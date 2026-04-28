//! Plain binary driver. Generates federated subgraph sets and operations
//! from a deterministic seed stream, runs each through the diff harness,
//! and prints aggregate stats. Saves a reproducer JSON for any divergence.
//!
//! Report modes:
//!   --report correctness   (default) — runs `run_diff`, saves divergences
//!                                       and panics, exit 1 on divergence
//!   --report perf          — builds both planners once per schema, times
//!                            each `plan()` per op with randomised
//!                            head/base order, prints distribution stats
//!   --report warm          — HEAD-only: plans each op N times through the
//!                            same planner, reports cold (1st) vs warm (2nd+)
//!                            timing + cache stats + allocation counts
//!   --report carryover     — HEAD-only: builds V1, plans ops to warm cache,
//!                            builds V2 with cache carryover (same or mutated
//!                            schema), asserts V2 plans match reference

use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use arbitrary::Unstructured;
use clap::Parser;
use clap::ValueEnum;
use serde_json::Value;

use apollo_federation_fuzz::compose::{ComposeOutcome, try_compose};
use apollo_federation_fuzz::diff::{normalize as normalize_plan, run_diff, DiffOutcome};
use apollo_federation_fuzz::harness::{CommonConfig, CommonOptions, PlannerHarness};
use apollo_federation_fuzz::op_gen::{generate_operation_with_config, OpGenConfig};
use apollo_federation_fuzz::preset_schemas::{PresetConfig, PresetSchema};
use apollo_federation_fuzz::subgraph_gen::{
    generate_federated_subgraphs, smoke_test_fixture, GenConfig, SubgraphSdl,
};
use apollo_federation_fuzz::{BasePlanner, HeadPlanner};

// ---------------------------------------------------------------------------
// Counting allocator — tracks bytes allocated between reset/snapshot calls.
// Zero overhead when not actively measuring (just an atomic increment).
// ---------------------------------------------------------------------------

struct CountingAllocator;

static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static TRACKING_ENABLED: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ENABLED.load(Ordering::Relaxed) != 0 {
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Set to true by the SIGINT handler. All loops check this and exit cleanly.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

fn should_stop() -> bool {
    SHUTDOWN.load(Ordering::Relaxed)
}

fn install_sigint_handler() {
    ctrlc::set_handler(move || {
        if SHUTDOWN.swap(true, Ordering::Relaxed) {
            // Second Ctrl-C: hard exit
            eprintln!("\nForced exit.");
            std::process::exit(130);
        }
        eprintln!("\nShutting down (Ctrl-C again to force)...");
    })
    .expect("failed to install Ctrl-C handler");
}

#[derive(Clone, Copy)]
struct AllocSnapshot {
    bytes: u64,
    count: u64,
}

fn alloc_start_tracking() {
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    TRACKING_ENABLED.store(1, Ordering::Relaxed);
}

fn alloc_snapshot() -> AllocSnapshot {
    AllocSnapshot {
        bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        count: ALLOC_COUNT.load(Ordering::Relaxed),
    }
}

fn alloc_stop_tracking() -> AllocSnapshot {
    TRACKING_ENABLED.store(0, Ordering::Relaxed);
    alloc_snapshot()
}

fn alloc_reset() {
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum PresetKind {
    /// Wide fan-out: N departments all extending Product with @requires.
    Wide,
    /// Deep chain: N subgraphs forming a linked list with crosscut @requires.
    Deep,
    /// Mesh: N subgraphs extending multiple core entities with cross-@requires.
    Mesh,
    /// Diamond: 10 domain subgraphs + N service subgraphs with multi-key entities,
    /// @provides, @requires chains, and diamond-shaped fetch dependencies.
    Diamond,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ReportMode {
    /// Diff plan output between HEAD and BASE; save reproducers on
    /// divergence or panic. Exit 1 if any divergence was found.
    Correctness,
    /// Time `plan()` on both sides; report distribution stats. Skips
    /// reproducer saves. Useful for "is HEAD faster than BASE?" runs.
    Perf,
    /// HEAD-only warm-cache measurement. Plans each operation multiple
    /// times through the same planner instance and reports cold (1st
    /// call) vs warm (subsequent) timing, allocation counts, and cache
    /// entry growth.
    Warm,
    /// HEAD-only cache carryover correctness. Builds V1 planner, plans
    /// ops to populate condition cache, builds V2 with carryover,
    /// asserts V2 plans match a fresh V2 (no carryover). Tests that
    /// carried-over cache entries don't corrupt plans.
    Carryover,
    /// Continuous schema evolution with cache carryover validation.
    /// Builds V1, plans ops, mutates schema to V2 (carrying over cache),
    /// plans more ops, mutates to V3 (carrying over again), etc.
    /// At each version, validates plans match a fresh planner.
    /// Tests long-lived cache correctness through many schema changes.
    HotSwap,
}

#[derive(Parser, Debug)]
#[command(about = "Differential fuzz of two apollo-federation query planners")]
struct Args {
    /// Total iterations to attempt. Set to 0 for unbounded (run until Ctrl-C).
    #[arg(long, default_value_t = 200)]
    iterations: u64,
    /// Deterministic seed. Set to 0 for a fixed seed, or use --random-seed
    /// for entropy from the OS.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Use a random seed from the OS instead of --seed. The chosen seed
    /// is printed at startup for reproducibility.
    #[arg(long, default_value_t = false)]
    random_seed: bool,
    /// Abort on the first divergence or panic (exit 1). Default in
    /// correctness mode is to continue and collect all findings.
    #[arg(long, default_value_t = false)]
    fail_fast: bool,
    /// Print progress stats every N iterations (0 = only at end).
    /// Default: every 500 iterations in bounded mode, every 100 in
    /// unbounded mode.
    #[arg(long)]
    progress_every: Option<u64>,
    /// How many planner runs to execute against each generated supergraph
    /// before regenerating. Set to 1 to maximize schema variety; raise to
    /// amortize the (relatively expensive) composition step.
    #[arg(long, default_value_t = 8)]
    ops_per_schema: u64,
    /// Use the hand-written 2-subgraph fixture instead of generating one.
    /// Useful for repro-debugging without schema noise.
    #[arg(long, default_value_t = false)]
    smoke_fixture: bool,
    /// Print one line per generated/skipped operation.
    #[arg(long, default_value_t = false)]
    verbose: bool,
    /// Save reproducers (subgraphs + supergraph + operation + diff) for any
    /// divergence under this directory.
    #[arg(long, default_value = "regressions")]
    regressions_dir: PathBuf,
    /// Enable `@defer`: op-gen sprinkles the directive on inline fragments
    /// and the planner is configured with `incremental_delivery = true` so
    /// `DeferNode` actually appears in plans.
    #[arg(long, default_value_t = false)]
    enable_defer: bool,
    /// Report mode. See ReportMode docs above.
    #[arg(long, value_enum, default_value_t = ReportMode::Correctness)]
    report: ReportMode,

    // --- Schema complexity knobs ---

    /// Minimum subgraphs per generated schema.
    #[arg(long)]
    min_subgraphs: Option<usize>,
    /// Maximum subgraphs per generated schema.
    #[arg(long)]
    max_subgraphs: Option<usize>,
    /// Minimum entity types per generated schema.
    #[arg(long)]
    min_entities: Option<usize>,
    /// Maximum entity types per generated schema.
    #[arg(long)]
    max_entities: Option<usize>,
    /// Maximum non-key fields per entity type.
    #[arg(long)]
    max_fields_per_entity: Option<usize>,
    /// Probability (0-255) that an entity gets a @requires link. Higher =
    /// more cross-subgraph condition edges = more condition cache activity.
    #[arg(long)]
    requires_chance: Option<u8>,
    /// Probability (0-255) for @provides on root fields.
    #[arg(long)]
    provides_chance: Option<u8>,
    /// Probability (0-255) for inter-entity reference fields.
    #[arg(long)]
    inter_entity_ref_chance: Option<u8>,
    /// Probability (0-255) for compound (multi-field) keys.
    #[arg(long)]
    compound_key_chance: Option<u8>,
    /// Probability (0-255) for a second @key on entities.
    #[arg(long)]
    multiple_key_chance: Option<u8>,

    // --- Warm-mode specific ---

    /// Number of times to re-plan each operation in warm mode (default 5).
    #[arg(long, default_value_t = 5)]
    warm_repeats: usize,

    // --- Carryover/hot-swap mode specific ---

    /// In carryover mode, mutate the schema between V1 and V2 (add a field).
    /// When false, V2 uses the identical schema — tests same-schema carryover.
    #[arg(long, default_value_t = false)]
    mutate_schema: bool,

    /// In hot-swap mode, how many schema versions to evolve through (default 10).
    #[arg(long, default_value_t = 10)]
    schema_versions: u64,

    /// In hot-swap mode, operations to plan per schema version (default 20).
    #[arg(long, default_value_t = 20)]
    ops_per_version: u64,

    // --- Preset schema mode ---

    /// Use a deterministic pathological schema preset instead of random
    /// generation. Overrides --smoke-fixture and schema complexity knobs.
    #[arg(long, value_enum)]
    preset: Option<PresetKind>,

    /// Number of extension subgraphs in preset schemas (default 50).
    #[arg(long, default_value_t = 50)]
    num_extensions: usize,

}

impl Args {
    fn gen_config(&self) -> GenConfig {
        let mut cfg = GenConfig::default();
        if let Some(v) = self.min_subgraphs {
            cfg.min_subgraphs = v;
        }
        if let Some(v) = self.max_subgraphs {
            cfg.max_subgraphs = v;
        }
        if let Some(v) = self.min_entities {
            cfg.min_entities = v;
        }
        if let Some(v) = self.max_entities {
            cfg.max_entities = v;
        }
        if let Some(v) = self.max_fields_per_entity {
            cfg.max_fields_per_entity = v;
        }
        if let Some(v) = self.requires_chance {
            cfg.requires_chance = v;
        }
        if let Some(v) = self.provides_chance {
            cfg.provides_chance = v;
        }
        if let Some(v) = self.inter_entity_ref_chance {
            cfg.inter_entity_ref_chance = v;
        }
        if let Some(v) = self.compound_key_chance {
            cfg.compound_key_chance = v;
        }
        if let Some(v) = self.multiple_key_chance {
            cfg.multiple_key_chance = v;
        }
        cfg
    }

    fn preset_config(&self) -> PresetConfig {
        PresetConfig {
            num_extensions: self.num_extensions,
        }
    }

    fn build_preset(&self) -> Option<PresetSchema> {
        let preset_cfg = self.preset_config();
        self.preset.map(|kind| match kind {
            PresetKind::Wide => apollo_federation_fuzz::preset_schemas::wide_fan_out(&preset_cfg),
            PresetKind::Deep => apollo_federation_fuzz::preset_schemas::deep_chain(&preset_cfg),
            PresetKind::Mesh => apollo_federation_fuzz::preset_schemas::mesh(&preset_cfg),
            PresetKind::Diamond => apollo_federation_fuzz::preset_schemas::diamond(&preset_cfg),
        })
    }

    fn effective_seed(&self) -> u64 {
        if self.random_seed {
            let mut buf = [0u8; 8];
            getrandom::fill(&mut buf).expect("failed to get random seed");
            u64::from_le_bytes(buf)
        } else {
            self.seed
        }
    }

    fn is_unbounded(&self) -> bool {
        self.iterations == 0
    }

    fn should_run(&self, i: u64) -> bool {
        !should_stop() && (self.is_unbounded() || i < self.iterations)
    }

    fn progress_interval(&self) -> u64 {
        self.progress_every.unwrap_or(if self.is_unbounded() { 100 } else { 500 })
    }
}

#[derive(Default, Debug)]
struct Stats {
    schemas_attempted: u64,
    schemas_composed: u64,
    schemas_compose_failed: u64,
    ops_attempted: u64,
    ops_skipped: u64,
    planned_identical: u64,
    planned_divergent: u64,
    planner_errored: u64,
    panicked: u64,
}

fn main() {
    install_sigint_handler();

    let args = Args::parse();
    let seed = args.effective_seed();
    if args.random_seed {
        eprintln!("random seed: {seed} (use --seed {seed} to reproduce)");
    }

    let cfg = CommonConfig {
        incremental_delivery: args.enable_defer,
        ..CommonConfig::default()
    };
    let opts = CommonOptions::default();
    let gen_cfg = args.gen_config();
    let op_cfg = OpGenConfig {
        defer_chance: if args.enable_defer { 64 } else { 0 },
        ..OpGenConfig::default()
    };

    match args.report {
        ReportMode::Perf => {
            run_perf_mode(args, seed, cfg, opts, gen_cfg, op_cfg);
            return;
        }
        ReportMode::Warm => {
            run_warm_mode(args, seed, cfg, opts, gen_cfg, op_cfg);
            return;
        }
        ReportMode::Carryover => {
            run_carryover_mode(args, seed, cfg, opts, gen_cfg, op_cfg);
            return;
        }
        ReportMode::HotSwap => {
            run_hot_swap_mode(args, seed, cfg, opts, gen_cfg, op_cfg);
            return;
        }
        ReportMode::Correctness => {}
    }

    if args.is_unbounded() {
        eprintln!("Running unbounded correctness fuzz (Ctrl-C to stop)...");
    }

    let start_time = Instant::now();
    let progress_interval = args.progress_interval();
    let mut stats = Stats::default();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut current_schema: Option<(Vec<SubgraphSdl>, String)> = None;
    let mut ops_done_for_schema: u64 = 0;
    let mut i: u64 = 0;

    while args.should_run(i) {
        let need_new_schema = args.smoke_fixture && current_schema.is_none()
            || (!args.smoke_fixture
                && (current_schema.is_none() || ops_done_for_schema >= args.ops_per_schema));

        if need_new_schema {
            let bytes = next_bytes(&mut state, 4096);
            let subgraphs = if args.smoke_fixture {
                smoke_test_fixture()
            } else {
                let mut u = Unstructured::new(&bytes);
                match generate_federated_subgraphs(&mut u, &gen_cfg) {
                    Ok(s) => s,
                    Err(e) => {
                        if args.verbose {
                            eprintln!("[{i}] gen subgraph err: {e}");
                        }
                        i += 1;
                        continue;
                    }
                }
            };
            stats.schemas_attempted += 1;
            match try_compose(&subgraphs) {
                ComposeOutcome::Composed { supergraph_sdl } => {
                    stats.schemas_composed += 1;
                    current_schema = Some((subgraphs, supergraph_sdl));
                    ops_done_for_schema = 0;
                }
                other => {
                    stats.schemas_compose_failed += 1;
                    if args.verbose {
                        eprintln!("[{i}] compose failed: {other:?}");
                    }
                    i += 1;
                    continue;
                }
            }
        }

        let Some((subgraphs, supergraph_sdl)) = current_schema.as_ref() else {
            i += 1;
            continue;
        };

        let op_bytes = next_bytes(&mut state, 1024);
        stats.ops_attempted += 1;
        let op_text = match generate_operation_with_config(supergraph_sdl, &op_bytes, &op_cfg) {
            Ok(op) => op,
            Err(e) => {
                stats.ops_skipped += 1;
                ops_done_for_schema += 1;
                if args.verbose {
                    eprintln!("[{i}] op skip: {e}");
                }
                i += 1;
                continue;
            }
        };

        let outcome = run_diff::<HeadPlanner, BasePlanner>(supergraph_sdl, &op_text, None, &cfg, &opts);
        ops_done_for_schema += 1;

        let mut is_finding = false;
        match outcome {
            DiffOutcome::Identical { .. } => stats.planned_identical += 1,
            DiffOutcome::Divergent {
                unified_diff,
                head: _,
                base: _,
            } => {
                stats.planned_divergent += 1;
                is_finding = true;
                let id = save_regression(
                    &args.regressions_dir,
                    i,
                    seed,
                    subgraphs,
                    supergraph_sdl,
                    &op_text,
                    &unified_diff,
                );
                println!("=== DIVERGENCE iter={i} saved={id} ===\n{unified_diff}");
            }
            DiffOutcome::EitherFailed { head, base } => {
                stats.planner_errored += 1;
                if args.verbose {
                    eprintln!(
                        "[{i}] planner err head_ok={} base_ok={}",
                        head.is_ok(),
                        base.is_ok()
                    );
                }
            }
            DiffOutcome::PanickedSide {
                head_panic,
                base_panic,
                ..
            } => {
                stats.panicked += 1;
                is_finding = true;
                let summary = format!("head_panic={head_panic:?}\nbase_panic={base_panic:?}\n",);
                let id = save_regression(
                    &args.regressions_dir,
                    i,
                    seed,
                    subgraphs,
                    supergraph_sdl,
                    &op_text,
                    &format!("=== PANIC ===\n{summary}"),
                );
                println!("=== PLANNER PANIC iter={i} saved={id} ===\n{summary}");
            }
        }

        if args.fail_fast && is_finding {
            eprintln!("--fail-fast: aborting after first finding at iter {i}");
            print_correctness_stats(&stats, seed, &start_time);
            std::process::exit(1);
        }

        // Periodic progress
        i += 1;
        if progress_interval > 0 && i % progress_interval == 0 {
            let elapsed = start_time.elapsed();
            let rate = i as f64 / elapsed.as_secs_f64();
            eprintln!(
                "[progress] iter={i} elapsed={:.0}s rate={rate:.0}/s schemas={}/{} \
                 identical={} diverged={} panicked={} errored={}",
                elapsed.as_secs_f64(),
                stats.schemas_composed,
                stats.schemas_attempted,
                stats.planned_identical,
                stats.planned_divergent,
                stats.panicked,
                stats.planner_errored,
            );
        }
    }

    if should_stop() {
        eprintln!("Interrupted after {i} iterations.");
    }

    print_correctness_stats(&stats, seed, &start_time);

    if stats.planned_divergent > 0 || stats.panicked > 0 {
        std::process::exit(1);
    }
}

fn print_correctness_stats(stats: &Stats, seed: u64, start_time: &Instant) {
    let elapsed = start_time.elapsed();
    println!("=== CORRECTNESS REPORT ===");
    println!("seed:    {seed}");
    println!("elapsed: {:.1}s", elapsed.as_secs_f64());
    println!(
        "schemas: attempted={} composed={} compose_failed={}",
        stats.schemas_attempted, stats.schemas_composed, stats.schemas_compose_failed
    );
    println!(
        "ops:     attempted={} skipped={} identical={} diverged={} panicked={} errored={}",
        stats.ops_attempted,
        stats.ops_skipped,
        stats.planned_identical,
        stats.planned_divergent,
        stats.panicked,
        stats.planner_errored,
    );
    if elapsed.as_secs() > 0 {
        let rate = stats.ops_attempted as f64 / elapsed.as_secs_f64();
        println!("rate:    {rate:.1} ops/s");
    }
    if stats.planned_divergent > 0 || stats.panicked > 0 {
        println!("FAIL: {} divergences, {} panics", stats.planned_divergent, stats.panicked);
    } else {
        println!("PASS: all plans identical");
    }
}

fn next_bytes(state: &mut u64, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(n);
    out
}

/// Saves a divergence/panic reproducer in the slim format consumed by
/// `tests/regression_replay.rs`.
fn save_regression(
    dir: &PathBuf,
    iter: u64,
    seed: u64,
    subgraphs: &[SubgraphSdl],
    supergraph_sdl: &str,
    op_text: &str,
    unified_diff: &str,
) -> String {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(supergraph_sdl);
    hasher.update(op_text);
    let id = hex::encode(&hasher.finalize()[..6]);

    let summary = summarize_finding(unified_diff);

    let _ = std::fs::create_dir_all(dir);
    let path = dir.join(format!("{id}.txt"));
    let mut content = String::new();
    content.push_str(&format!("iter={iter} seed={seed} sha1_prefix={id}\n"));
    content.push_str(&format!("summary: {summary}\n\n"));
    content.push_str("=== SUBGRAPHS ===\n");
    for s in subgraphs {
        content.push_str(&format!("# {}\n{}\n", s.name, s.sdl));
    }
    content.push_str("\n=== OPERATION ===\n");
    content.push_str(op_text);
    let _ = std::fs::write(path, content);
    id
}

fn summarize_finding(unified_diff: &str) -> String {
    if let Some(rest) = unified_diff.strip_prefix("=== PANIC ===\n") {
        for line in rest.lines() {
            if let Some(start) = line.find("=Some(\"") {
                let inner = &line[start + 7..];
                if let Some(end) = inner.find("\")") {
                    return format!("PANIC: {}", &inner[..end]);
                }
            }
        }
        return "PANIC".to_string();
    }
    let mut tags: Vec<&'static str> = Vec::new();
    if unified_diff.contains("on Query") {
        tags.push("PR #7580");
    }
    if unified_diff.contains("\"Condition\":") {
        tags.push("FED-505 Condition");
    }
    if unified_diff
        .lines()
        .any(|l| l.starts_with('+') && l.contains("\"sub_selection\":"))
    {
        tags.push("defer sub_selection (C)");
    }
    let has_minus_field_str = unified_diff
        .lines()
        .any(|l| l.starts_with('-') && l.contains("\"Field\": \""));
    let has_plus_field_obj = unified_diff
        .lines()
        .any(|l| l.starts_with('+') && l.contains("\"Field\": {"));
    if has_minus_field_str && has_plus_field_obj {
        tags.push("defer Field repr (D)");
    }
    if tags.is_empty() {
        "(uncategorised diff — possible Class E)".to_string()
    } else {
        tags.join(" + ")
    }
}

// =========================================================================
// Perf-report mode (HEAD vs BASE wall-clock, existing)
// =========================================================================

#[derive(Default, Debug)]
struct PerfStats {
    schemas_attempted: u64,
    schemas_composed: u64,
    schemas_built: u64,
    ops_attempted: u64,
    ops_skipped: u64,
    ops_diverged: u64,
    ops_panicked: u64,
    ops_errored: u64,
    samples: Vec<(u128, u128)>,
    shapes: Vec<(usize, usize, usize, usize)>,
}

fn run_perf_mode(args: Args, seed: u64, cfg: CommonConfig, opts: CommonOptions, gen_cfg: GenConfig, op_cfg: OpGenConfig) {
    let mut stats = PerfStats::default();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);

    struct Planners {
        head: HeadPlanner,
        base: BasePlanner,
    }
    let mut current: Option<Planners> = None;
    let mut current_supergraph: Option<String> = None;
    let mut ops_done_for_schema: u64 = 0;

    let mut i: u64 = 0;
    while args.should_run(i) {
        let need_new_schema = args.smoke_fixture && current.is_none()
            || (!args.smoke_fixture
                && (current.is_none() || ops_done_for_schema >= args.ops_per_schema));

        if need_new_schema {
            let bytes = next_bytes(&mut state, 4096);
            let subgraphs = if args.smoke_fixture {
                smoke_test_fixture()
            } else {
                let mut u = Unstructured::new(&bytes);
                match generate_federated_subgraphs(&mut u, &gen_cfg) {
                    Ok(s) => s,
                    Err(_) => { i += 1; continue; },
                }
            };
            stats.schemas_attempted += 1;
            let supergraph_sdl = match try_compose(&subgraphs) {
                ComposeOutcome::Composed { supergraph_sdl } => {
                    stats.schemas_composed += 1;
                    supergraph_sdl
                }
                _ => { i += 1; continue; },
            };
            let head = match HeadPlanner::build(&supergraph_sdl, &cfg) {
                Ok(p) => p,
                Err(_) => { i += 1; continue; },
            };
            let base = match BasePlanner::build(&supergraph_sdl, &cfg) {
                Ok(p) => p,
                Err(_) => { i += 1; continue; },
            };
            stats.schemas_built += 1;
            current = Some(Planners { head, base });
            current_supergraph = Some(supergraph_sdl);
            ops_done_for_schema = 0;
        }

        let Some(planners) = current.as_ref() else {
            i += 1;
            continue;
        };
        let Some(supergraph_sdl) = current_supergraph.as_ref() else {
            i += 1;
            continue;
        };

        let op_bytes = next_bytes(&mut state, 1024);
        stats.ops_attempted += 1;
        let op_text = match generate_operation_with_config(supergraph_sdl, &op_bytes, &op_cfg) {
            Ok(op) => op,
            Err(_) => {
                stats.ops_skipped += 1;
                ops_done_for_schema += 1;
                i += 1;
                continue;
            }
        };

        let head_first = (next_bytes(&mut state, 1)[0] & 1) == 0;
        let plan_h = || {
            let t = Instant::now();
            let r = catch_unwind(AssertUnwindSafe(|| planners.head.plan(&op_text, None, &opts)));
            (t.elapsed().as_micros(), r)
        };
        let plan_b = || {
            let t = Instant::now();
            let r = catch_unwind(AssertUnwindSafe(|| planners.base.plan(&op_text, None, &opts)));
            (t.elapsed().as_micros(), r)
        };
        let ((head_us, head_res), (base_us, base_res)) = if head_first {
            let h = plan_h();
            let b = plan_b();
            (h, b)
        } else {
            let b = plan_b();
            let h = plan_h();
            (h, b)
        };

        ops_done_for_schema += 1;

        let (head_plan, base_plan) = match (head_res, base_res) {
            (Ok(Ok(h)), Ok(Ok(b))) => (h, b),
            (Err(_), _) | (_, Err(_)) => {
                stats.ops_panicked += 1;
                i += 1;
                continue;
            }
            _ => {
                stats.ops_errored += 1;
                i += 1;
                continue;
            }
        };

        if !plans_equivalent(&head_plan, &base_plan) {
            stats.ops_diverged += 1;
            i += 1;
            continue;
        }

        let h_bytes = serde_json::to_string(&head_plan)
            .map(|s| s.len())
            .unwrap_or(0);
        let b_bytes = serde_json::to_string(&base_plan)
            .map(|s| s.len())
            .unwrap_or(0);
        let h_fetch = count_fetch_nodes(&head_plan);
        let b_fetch = count_fetch_nodes(&base_plan);

        stats.samples.push((head_us, base_us));
        stats.shapes.push((h_fetch, b_fetch, h_bytes, b_bytes));

        if args.verbose {
            eprintln!(
                "[{i}] head={head_us}us base={base_us}us h_fetch={h_fetch} b_fetch={b_fetch} ratio={:.2}",
                head_us as f64 / base_us.max(1) as f64
            );
        }
        i += 1;
    }

    print_perf_report(&stats);
}

fn plans_equivalent(a: &Value, b: &Value) -> bool {
    normalize_plan(a) == normalize_plan(b)
}

fn count_fetch_nodes(v: &Value) -> usize {
    match v {
        Value::Object(map) => {
            let mut n = 0;
            for (k, vv) in map {
                if k == "Fetch" {
                    n += 1;
                }
                n += count_fetch_nodes(vv);
            }
            n
        }
        Value::Array(arr) => arr.iter().map(count_fetch_nodes).sum(),
        _ => 0,
    }
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn percentile_f64(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_perf_report(stats: &PerfStats) {
    println!("=== PERF REPORT ===");
    println!(
        "schemas: attempted={} composed={} built={}",
        stats.schemas_attempted, stats.schemas_composed, stats.schemas_built
    );
    println!(
        "ops:     attempted={} skipped={} diverged={} panicked={} errored={} sampled={}",
        stats.ops_attempted,
        stats.ops_skipped,
        stats.ops_diverged,
        stats.ops_panicked,
        stats.ops_errored,
        stats.samples.len()
    );

    if stats.samples.is_empty() {
        println!("(no perf samples)");
        return;
    }

    let mut head_us: Vec<u128> = stats.samples.iter().map(|(h, _)| *h).collect();
    let mut base_us: Vec<u128> = stats.samples.iter().map(|(_, b)| *b).collect();
    head_us.sort();
    base_us.sort();

    let mut ratios: Vec<f64> = stats
        .samples
        .iter()
        .map(|(h, b)| *h as f64 / (*b).max(1) as f64)
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("plan() wall-clock micros:");
    println!(
        "  head: min={:>6} p50={:>7} p95={:>8} p99={:>8} max={:>9}",
        head_us[0],
        percentile(&head_us, 0.5),
        percentile(&head_us, 0.95),
        percentile(&head_us, 0.99),
        head_us[head_us.len() - 1]
    );
    println!(
        "  base: min={:>6} p50={:>7} p95={:>8} p99={:>8} max={:>9}",
        base_us[0],
        percentile(&base_us, 0.5),
        percentile(&base_us, 0.95),
        percentile(&base_us, 0.99),
        base_us[base_us.len() - 1]
    );

    println!();
    println!("ratio (head / base):");
    println!(
        "  min={:.2} p50={:.2} p95={:.2} p99={:.2} max={:.2}",
        ratios[0],
        percentile_f64(&ratios, 0.5),
        percentile_f64(&ratios, 0.95),
        percentile_f64(&ratios, 0.99),
        ratios[ratios.len() - 1]
    );
    let median = percentile_f64(&ratios, 0.5);
    println!(
        "  verdict (median ratio): {}",
        if median < 0.95 {
            format!("HEAD faster (median {median:.2}x base)")
        } else if median > 1.05 {
            format!("HEAD slower (median {median:.2}x base)")
        } else {
            format!("within +/-5% (median {median:.2}x base) -- likely noise")
        }
    );

    let mut h_fetch: Vec<usize> = stats.shapes.iter().map(|s| s.0).collect();
    let mut b_fetch: Vec<usize> = stats.shapes.iter().map(|s| s.1).collect();
    let mut h_bytes: Vec<usize> = stats.shapes.iter().map(|s| s.2).collect();
    let mut b_bytes: Vec<usize> = stats.shapes.iter().map(|s| s.3).collect();
    h_fetch.sort();
    b_fetch.sort();
    h_bytes.sort();
    b_bytes.sort();
    let median_usize = |v: &[usize]| -> usize { v[v.len() / 2] };

    println!();
    println!("plan-shape (median across sampled ops):");
    println!(
        "  fetch nodes: head={} base={}",
        median_usize(&h_fetch),
        median_usize(&b_fetch)
    );
    println!(
        "  json bytes:  head={} base={}",
        median_usize(&h_bytes),
        median_usize(&b_bytes)
    );
}

// =========================================================================
// Warm-cache report mode (HEAD-only: cold vs warm measurement)
// =========================================================================

struct WarmSample {
    cold_us: u128,
    cold_alloc_bytes: u64,
    cold_alloc_count: u64,
    warm_us: Vec<u128>,
    warm_alloc_bytes: Vec<u64>,
    warm_alloc_count: Vec<u64>,
    cache_entries_after: usize,
    fetch_count: usize,
}

fn run_warm_mode(
    args: Args,
    seed: u64,
    cfg: CommonConfig,
    opts: CommonOptions,
    gen_cfg: GenConfig,
    op_cfg: OpGenConfig,
) {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut samples: Vec<WarmSample> = Vec::new();
    let mut schemas_attempted: u64 = 0;
    let mut schemas_composed: u64 = 0;
    let mut schemas_built: u64 = 0;
    let mut ops_attempted: u64 = 0;
    let mut ops_skipped: u64 = 0;
    let mut ops_errored: u64 = 0;

    let mut current_planner: Option<HeadPlanner> = None;
    let mut current_supergraph: Option<String> = None;
    let mut ops_done_for_schema: u64 = 0;
    let progress_interval = args.progress_interval();
    let start_time = Instant::now();

    // If a preset is specified, compose it once up front and use it as
    // the fixed schema for the entire run. The schema never rotates —
    // only the random operations change. This is the normal way to run
    // presets: the schema provides pathological structure, the fuzzer
    // provides query variety.
    if let Some(preset) = args.build_preset() {
        eprintln!(
            "Composing preset schema ({} subgraphs)...",
            preset.subgraphs.len(),
        );
        let supergraph_sdl = match try_compose(&preset.subgraphs) {
            ComposeOutcome::Composed { supergraph_sdl } => supergraph_sdl,
            ComposeOutcome::ParseFailed { errors } => {
                eprintln!("Preset subgraph parse failed:");
                for e in &errors { eprintln!("  {e}"); }
                std::process::exit(1);
            }
            ComposeOutcome::CompositionFailed { errors } => {
                eprintln!("Preset composition failed:");
                for e in &errors { eprintln!("  {e}"); }
                std::process::exit(1);
            }
        };
        eprintln!(
            "Composed: {} bytes. Building planner...",
            supergraph_sdl.len()
        );
        let planner = match HeadPlanner::build(&supergraph_sdl, &cfg) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to build planner: {e}");
                std::process::exit(1);
            }
        };
        schemas_attempted = 1;
        schemas_composed = 1;
        schemas_built = 1;

        // Run preset stress queries first (these are hand-crafted to hit
        // the worst-case patterns: all-service queries, cross-entity traversals)
        if !preset.stress_queries.is_empty() {
            eprintln!(
                "Running {} stress queries x {} warm repeats...",
                preset.stress_queries.len(),
                args.warm_repeats,
            );
            for (qi, pq) in preset.stress_queries.iter().enumerate() {
                if should_stop() { break; }
                eprintln!("  query {:>2}: {} ...", qi, pq.name);
                ops_attempted += 1;
                let sample = measure_warm_sample(
                    &planner,
                    &pq.query,
                    &opts,
                    args.warm_repeats,
                    true,
                    qi as u64,
                );
                match sample {
                    Some(s) => samples.push(s),
                    None => {
                        ops_errored += 1;
                        eprintln!("    ERROR: planning failed for {}", pq.name);
                    }
                }
            }
            eprintln!("Running random ops against preset schema (Ctrl-C to stop)...");
        }

        current_planner = Some(planner);
        current_supergraph = Some(supergraph_sdl);
        // Never rotate — ops_per_schema is effectively infinite for presets
    }

    let mut i: u64 = 0;
    while args.should_run(i) {
        // Schema rotation (only for random-gen mode, not presets)
        if args.preset.is_none() {
            let need_new_schema = args.smoke_fixture && current_planner.is_none()
                || (!args.smoke_fixture
                    && (current_planner.is_none() || ops_done_for_schema >= args.ops_per_schema));

            if need_new_schema {
                let bytes = next_bytes(&mut state, 4096);
                let subgraphs = if args.smoke_fixture {
                    smoke_test_fixture()
                } else {
                    let mut u = Unstructured::new(&bytes);
                    match generate_federated_subgraphs(&mut u, &gen_cfg) {
                        Ok(s) => s,
                        Err(_) => { i += 1; continue; },
                    }
                };
                schemas_attempted += 1;
                let supergraph_sdl = match try_compose(&subgraphs) {
                    ComposeOutcome::Composed { supergraph_sdl } => {
                        schemas_composed += 1;
                        supergraph_sdl
                    }
                    _ => { i += 1; continue; },
                };
                let planner = match HeadPlanner::build(&supergraph_sdl, &cfg) {
                    Ok(p) => p,
                    Err(_) => { i += 1; continue; },
                };
                schemas_built += 1;
                current_planner = Some(planner);
                current_supergraph = Some(supergraph_sdl);
                ops_done_for_schema = 0;
            }
        }

        let Some(planner) = current_planner.as_ref() else {
            i += 1;
            continue;
        };
        let Some(supergraph_sdl) = current_supergraph.as_ref() else {
            i += 1;
            continue;
        };

        let op_bytes = next_bytes(&mut state, 1024);
        ops_attempted += 1;
        let op_text = match generate_operation_with_config(supergraph_sdl, &op_bytes, &op_cfg) {
            Ok(op) => op,
            Err(_) => {
                ops_skipped += 1;
                ops_done_for_schema += 1;
                i += 1;
                continue;
            }
        };

        let sample = measure_warm_sample(planner, &op_text, &opts, args.warm_repeats, args.verbose, i);
        match sample {
            Some(s) => samples.push(s),
            None => ops_errored += 1,
        }

        ops_done_for_schema += 1;
        i += 1;

        if progress_interval > 0 && i % progress_interval == 0 {
            let elapsed = start_time.elapsed();
            let rate = i as f64 / elapsed.as_secs_f64();
            eprintln!(
                "[progress] iter={i} elapsed={:.0}s rate={rate:.0}/s \
                 sampled={} skipped={ops_skipped} errored={ops_errored} \
                 cache={}",
                elapsed.as_secs_f64(),
                samples.len(),
                current_planner.as_ref().map_or(0, |p| p.condition_resolver_cache_len()),
            );
        }
    }

    print_warm_report(
        &samples,
        schemas_attempted,
        schemas_composed,
        schemas_built,
        ops_attempted,
        ops_skipped,
        ops_errored,
    );
}

/// Measure cold-vs-warm planning for a single operation through a planner.
fn measure_warm_sample(
    planner: &HeadPlanner,
    op_text: &str,
    opts: &CommonOptions,
    warm_repeats: usize,
    verbose: bool,
    idx: u64,
) -> Option<WarmSample> {
    // Cold plan (1st call)
    alloc_start_tracking();
    let cold_start = Instant::now();
    let cold_result =
        catch_unwind(AssertUnwindSafe(|| planner.plan(op_text, None, opts)));
    let cold_us = cold_start.elapsed().as_micros();
    let cold_alloc = alloc_stop_tracking();

    let cold_plan = match cold_result {
        Ok(Ok(p)) => p,
        _ => return None,
    };

    // Warm plans (repeats 2..N)
    let mut warm_us = Vec::with_capacity(warm_repeats);
    let mut warm_alloc_bytes = Vec::with_capacity(warm_repeats);
    let mut warm_alloc_count = Vec::with_capacity(warm_repeats);
    let mut all_match = true;

    for _ in 0..warm_repeats {
        alloc_start_tracking();
        let start = Instant::now();
        let result =
            catch_unwind(AssertUnwindSafe(|| planner.plan(op_text, None, opts)));
        let us = start.elapsed().as_micros();
        let alloc = alloc_stop_tracking();

        match result {
            Ok(Ok(ref p)) => {
                if normalize_plan(&cold_plan) != normalize_plan(p) {
                    all_match = false;
                }
                warm_us.push(us);
                warm_alloc_bytes.push(alloc.bytes);
                warm_alloc_count.push(alloc.count);
            }
            _ => break,
        }
    }

    if !all_match {
        eprintln!("[{idx}] WARNING: warm replay produced different plan!");
    }

    let cache_entries = planner.condition_resolver_cache_len();
    let fetch_count = count_fetch_nodes(&cold_plan);

    if verbose {
        let warm_median = if warm_us.is_empty() {
            0
        } else {
            let mut sorted = warm_us.clone();
            sorted.sort();
            sorted[sorted.len() / 2]
        };
        let speedup = if warm_median > 0 {
            cold_us as f64 / warm_median as f64
        } else {
            0.0
        };
        eprintln!(
            "    [{idx}] cold={cold_us}us warm_median={warm_median}us speedup={speedup:.1}x \
             cache={cache_entries} fetches={fetch_count} \
             cold_alloc={}KB warm_alloc={}KB",
            cold_alloc.bytes / 1024,
            warm_alloc_bytes.first().unwrap_or(&0) / 1024,
        );
    }

    Some(WarmSample {
        cold_us,
        cold_alloc_bytes: cold_alloc.bytes,
        cold_alloc_count: cold_alloc.count,
        warm_us,
        warm_alloc_bytes,
        warm_alloc_count,
        cache_entries_after: cache_entries,
        fetch_count,
    })
}

fn print_warm_report(
    samples: &[WarmSample],
    schemas_attempted: u64,
    schemas_composed: u64,
    schemas_built: u64,
    ops_attempted: u64,
    ops_skipped: u64,
    ops_errored: u64,
) {
    println!("=== WARM-CACHE REPORT ===");
    println!(
        "schemas: attempted={schemas_attempted} composed={schemas_composed} built={schemas_built}"
    );
    println!(
        "ops:     attempted={ops_attempted} skipped={ops_skipped} errored={ops_errored} sampled={}",
        samples.len()
    );

    if samples.is_empty() {
        println!("(no samples)");
        return;
    }

    // Cold timing
    let mut cold_us: Vec<u128> = samples.iter().map(|s| s.cold_us).collect();
    cold_us.sort();

    // Warm timing: use median of each sample's warm repeats
    let mut warm_median_us: Vec<u128> = samples
        .iter()
        .filter_map(|s| {
            if s.warm_us.is_empty() {
                return None;
            }
            let mut sorted = s.warm_us.clone();
            sorted.sort();
            Some(sorted[sorted.len() / 2])
        })
        .collect();
    warm_median_us.sort();

    // Speedup ratios: cold / warm_median
    let mut speedups: Vec<f64> = samples
        .iter()
        .filter_map(|s| {
            if s.warm_us.is_empty() {
                return None;
            }
            let mut sorted = s.warm_us.clone();
            sorted.sort();
            let warm_med = sorted[sorted.len() / 2];
            if warm_med == 0 {
                return None;
            }
            Some(s.cold_us as f64 / warm_med as f64)
        })
        .collect();
    speedups.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("plan() wall-clock micros:");
    println!(
        "  cold:        min={:>6} p50={:>7} p95={:>8} max={:>9}",
        cold_us[0],
        percentile(&cold_us, 0.5),
        percentile(&cold_us, 0.95),
        cold_us[cold_us.len() - 1]
    );
    if !warm_median_us.is_empty() {
        println!(
            "  warm(median): min={:>6} p50={:>7} p95={:>8} max={:>9}",
            warm_median_us[0],
            percentile(&warm_median_us, 0.5),
            percentile(&warm_median_us, 0.95),
            warm_median_us[warm_median_us.len() - 1]
        );
    }

    if !speedups.is_empty() {
        println!();
        println!("cold/warm speedup ratio:");
        println!(
            "  min={:.2}x p50={:.2}x p95={:.2}x max={:.2}x",
            speedups[0],
            percentile_f64(&speedups, 0.5),
            percentile_f64(&speedups, 0.95),
            speedups[speedups.len() - 1]
        );
    }

    // Allocation stats
    let mut cold_alloc_kb: Vec<u64> = samples.iter().map(|s| s.cold_alloc_bytes / 1024).collect();
    cold_alloc_kb.sort();
    let mut warm_alloc_kb: Vec<u64> = samples
        .iter()
        .filter_map(|s| s.warm_alloc_bytes.first().copied())
        .map(|b| b / 1024)
        .collect();
    warm_alloc_kb.sort();

    let mut alloc_savings: Vec<f64> = samples
        .iter()
        .filter_map(|s| {
            let warm_b = *s.warm_alloc_bytes.first()?;
            if s.cold_alloc_bytes == 0 {
                return None;
            }
            Some(1.0 - (warm_b as f64 / s.cold_alloc_bytes as f64))
        })
        .collect();
    alloc_savings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("allocations (KB):");
    println!(
        "  cold:  min={:>6} p50={:>7} p95={:>8} max={:>9}",
        cold_alloc_kb[0],
        percentile_u64(&cold_alloc_kb, 0.5),
        percentile_u64(&cold_alloc_kb, 0.95),
        cold_alloc_kb[cold_alloc_kb.len() - 1]
    );
    if !warm_alloc_kb.is_empty() {
        println!(
            "  warm:  min={:>6} p50={:>7} p95={:>8} max={:>9}",
            warm_alloc_kb[0],
            percentile_u64(&warm_alloc_kb, 0.5),
            percentile_u64(&warm_alloc_kb, 0.95),
            warm_alloc_kb[warm_alloc_kb.len() - 1]
        );
    }
    if !alloc_savings.is_empty() {
        println!(
            "  warm savings: p50={:.0}% p95={:.0}%",
            percentile_f64(&alloc_savings, 0.5) * 100.0,
            percentile_f64(&alloc_savings, 0.95) * 100.0,
        );
    }

    // Cache stats
    let mut cache_entries: Vec<usize> = samples.iter().map(|s| s.cache_entries_after).collect();
    cache_entries.sort();
    let max_cache = *cache_entries.last().unwrap_or(&0);
    let with_cache = cache_entries.iter().filter(|&&c| c > 0).count();

    println!();
    println!("condition cache:");
    println!(
        "  ops with cache entries: {}/{} ({:.0}%)",
        with_cache,
        samples.len(),
        with_cache as f64 / samples.len() as f64 * 100.0
    );
    println!("  max entries: {max_cache}");

    // Fetch count distribution
    let mut fetches: Vec<usize> = samples.iter().map(|s| s.fetch_count).collect();
    fetches.sort();
    println!();
    println!("plan complexity:");
    println!(
        "  fetch nodes: min={} p50={} p95={} max={}",
        fetches[0],
        fetches[fetches.len() / 2],
        fetches[(fetches.len() as f64 * 0.95) as usize],
        fetches[fetches.len() - 1]
    );
}

fn percentile_u64(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// =========================================================================
// Carryover correctness mode (HEAD-only: V1 -> V2 with cache)
// =========================================================================

fn run_carryover_mode(
    args: Args,
    seed: u64,
    cfg: CommonConfig,
    opts: CommonOptions,
    gen_cfg: GenConfig,
    op_cfg: OpGenConfig,
) {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut schemas_attempted: u64 = 0;
    let mut schemas_composed: u64 = 0;
    let mut schemas_tested: u64 = 0;
    let mut ops_tested: u64 = 0;
    let mut ops_skipped: u64 = 0;
    let mut mismatches: u64 = 0;
    let mut cache_entries_total: u64 = 0;

    for _schema_round in 0.. {
        if should_stop() || (!args.is_unbounded() && ops_tested + ops_skipped >= args.iterations) {
            break;
        }

        // Generate schema
        let bytes = next_bytes(&mut state, 4096);
        let subgraphs = if args.smoke_fixture {
            smoke_test_fixture()
        } else {
            let mut u = Unstructured::new(&bytes);
            match generate_federated_subgraphs(&mut u, &gen_cfg) {
                Ok(s) => s,
                Err(_) => continue,
            }
        };
        schemas_attempted += 1;

        let supergraph_sdl = match try_compose(&subgraphs) {
            ComposeOutcome::Composed { supergraph_sdl } => {
                schemas_composed += 1;
                supergraph_sdl
            }
            _ => continue,
        };

        // Build V1 planner
        let planner_v1 = match HeadPlanner::build(&supergraph_sdl, &cfg) {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Generate and plan ops through V1 to populate cache
        let mut v1_plans: Vec<(String, Value)> = Vec::new();
        for _ in 0..args.ops_per_schema {
            let op_bytes = next_bytes(&mut state, 1024);
            let op_text = match generate_operation_with_config(&supergraph_sdl, &op_bytes, &op_cfg)
            {
                Ok(op) => op,
                Err(_) => {
                    ops_skipped += 1;
                    continue;
                }
            };
            match planner_v1.plan(&op_text, None, &opts) {
                Ok(plan) => v1_plans.push((op_text, plan)),
                Err(_) => {
                    ops_skipped += 1;
                }
            }
        }

        if v1_plans.is_empty() {
            continue;
        }

        let v1_cache_len = planner_v1.condition_resolver_cache_len();
        cache_entries_total += v1_cache_len as u64;

        // Determine the V2 schema (same or mutated)
        let v2_sdl = if args.mutate_schema {
            match mutate_supergraph_add_field(&supergraph_sdl, 0) {
                Some(s) => s,
                None => supergraph_sdl.clone(), // Fall back to same schema
            }
        } else {
            supergraph_sdl.clone()
        };

        // Build V2-fresh (no carryover) — the reference
        let planner_v2_fresh = match HeadPlanner::build(&v2_sdl, &cfg) {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Build V2-carryover (with V1's cache)
        let planner_v2_carryover = match HeadPlanner::build_with_previous_cache(
            &v2_sdl,
            &cfg,
            planner_v1.condition_resolver_cache(),
        ) {
            Ok(p) => p,
            Err(e) => {
                if args.verbose {
                    eprintln!("V2-carryover build failed: {e}");
                }
                continue;
            }
        };

        schemas_tested += 1;

        // Re-plan all ops through both V2 planners and compare
        for (op_text, _v1_plan) in &v1_plans {
            let fresh_plan = match planner_v2_fresh.plan(op_text, None, &opts) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let carryover_plan = match planner_v2_carryover.plan(op_text, None, &opts) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("CARRYOVER PLAN FAILED: {e}\n  op: {op_text}");
                    mismatches += 1;
                    ops_tested += 1;
                    continue;
                }
            };

            if normalize_plan(&fresh_plan) != normalize_plan(&carryover_plan) {
                mismatches += 1;
                eprintln!(
                    "CARRYOVER MISMATCH (schema_round={_schema_round} cache_len={v1_cache_len} mutated={}):",
                    args.mutate_schema
                );
                eprintln!("  op: {op_text}");
                // Print a short diff
                let fresh_str = serde_json::to_string_pretty(&fresh_plan).unwrap_or_default();
                let carryover_str =
                    serde_json::to_string_pretty(&carryover_plan).unwrap_or_default();
                for change in similar::TextDiff::from_lines(&fresh_str, &carryover_str)
                    .iter_all_changes()
                {
                    let sign = match change.tag() {
                        similar::ChangeTag::Delete => "-",
                        similar::ChangeTag::Insert => "+",
                        similar::ChangeTag::Equal => continue,
                    };
                    eprint!("  {sign}{change}");
                }
                eprintln!();
            }

            ops_tested += 1;
        }

        if args.verbose {
            eprintln!(
                "[schema {_schema_round}] subgraphs={} v1_cache={v1_cache_len} ops={} mutated={}",
                subgraphs.len(),
                v1_plans.len(),
                args.mutate_schema
            );
        }
    }

    println!("=== CARRYOVER CORRECTNESS REPORT ===");
    println!(
        "schemas: attempted={schemas_attempted} composed={schemas_composed} tested={schemas_tested}"
    );
    println!(
        "ops:     tested={ops_tested} skipped={ops_skipped} mismatches={mismatches}"
    );
    println!("cache:   total entries imported={cache_entries_total}");
    println!("schema mutation: {}", if args.mutate_schema { "enabled" } else { "disabled" });
    println!();

    if mismatches > 0 {
        println!("FAIL: {mismatches} carryover mismatches found");
        std::process::exit(1);
    } else {
        println!("PASS: all carryover plans match fresh plans");
    }
}

// =========================================================================
// Hot-swap mode (continuous schema evolution with cache carryover)
// =========================================================================

fn run_hot_swap_mode(
    args: Args,
    seed: u64,
    cfg: CommonConfig,
    opts: CommonOptions,
    gen_cfg: GenConfig,
    op_cfg: OpGenConfig,
) {
    let mut state = seed.wrapping_add(0xB0BA_CAFE_DEAD_BEEF);
    let start_time = Instant::now();
    let progress_interval = args.progress_interval();

    let mut total_versions: u64 = 0;
    let mut total_ops: u64 = 0;
    let mut total_skipped: u64 = 0;
    let mut total_mismatches: u64 = 0;
    let mut total_rounds: u64 = 0;
    let mut max_cache_entries: usize = 0;
    let mut mutation_counts = [0u64; 4]; // add_field, rename_field, add_type, identity

    // Outer loop: each round starts from a fresh schema and evolves it
    while !should_stop() && (args.is_unbounded() || total_rounds < args.iterations) {
        total_rounds += 1;

        // Generate initial schema (V1)
        let (_initial_subgraphs, initial_sdl) = if let Some(preset) = args.build_preset() {
            let sdl = match try_compose(&preset.subgraphs) {
                ComposeOutcome::Composed { supergraph_sdl } => supergraph_sdl,
                _ => continue,
            };
            (preset.subgraphs, sdl)
        } else {
            let bytes = next_bytes(&mut state, 4096);
            let subgraphs = if args.smoke_fixture {
                smoke_test_fixture()
            } else {
                let mut u = Unstructured::new(&bytes);
                match generate_federated_subgraphs(&mut u, &gen_cfg) {
                    Ok(s) => s,
                    Err(_) => continue,
                }
            };
            let sdl = match try_compose(&subgraphs) {
                ComposeOutcome::Composed { supergraph_sdl } => supergraph_sdl,
                _ => continue,
            };
            (subgraphs, sdl)
        };

        // Build V1 planner
        let mut current_planner = match HeadPlanner::build(&initial_sdl, &cfg) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let mut current_sdl = initial_sdl;
        total_versions += 1;

        // Generate a pool of operations once from V1's API schema.
        // These operations are replayed after each schema mutation.
        // Operations that become invalid after a mutation are skipped.
        let mut op_pool: Vec<String> = Vec::new();
        for _ in 0..args.ops_per_version {
            let op_bytes = next_bytes(&mut state, 1024);
            if let Ok(op) = generate_operation_with_config(&current_sdl, &op_bytes, &op_cfg) {
                op_pool.push(op);
            }
        }

        // Plan all ops through V1 to warm the cache
        for op in &op_pool {
            let _ = current_planner.plan(op, None, &opts);
        }

        // Evolve through schema_versions mutations
        for version in 1..=args.schema_versions {
            if should_stop() {
                break;
            }

            // Pick a mutation type based on the version number for variety
            let mutation_kind = (state % 4) as usize;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

            let mutated_sdl = match mutation_kind {
                0 => {
                    mutation_counts[0] += 1;
                    mutate_supergraph_add_field(&current_sdl, version)
                }
                1 => {
                    mutation_counts[1] += 1;
                    mutate_supergraph_rename_field(&current_sdl, version)
                }
                2 => {
                    mutation_counts[2] += 1;
                    mutate_supergraph_add_type(&current_sdl, version)
                }
                _ => {
                    // Identity mutation — same schema, tests pure carryover
                    mutation_counts[3] += 1;
                    Some(current_sdl.clone())
                }
            };

            let next_sdl = match mutated_sdl {
                Some(sdl) => sdl,
                None => current_sdl.clone(), // fall back to identity
            };

            // Build fresh planner (reference — no carryover)
            let t_build = Instant::now();
            let planner_fresh = match HeadPlanner::build(&next_sdl, &cfg) {
                Ok(p) => p,
                Err(_) => {
                    // Mutation produced invalid schema; skip this version
                    continue;
                }
            };
            let fresh_build_ms = t_build.elapsed().as_millis();

            // Build carryover planner (with previous version's cache)
            let t_carry = Instant::now();
            let planner_carryover = match HeadPlanner::build_with_previous_cache(
                &next_sdl,
                &cfg,
                current_planner.condition_resolver_cache(),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let carry_build_ms = t_carry.elapsed().as_millis();

            total_versions += 1;

            if args.verbose {
                eprintln!(
                    "  [v{version}] build: fresh={fresh_build_ms}ms carry={carry_build_ms}ms sdl={}KB",
                    next_sdl.len() / 1024,
                );
            }

            // Generate some new operations for this schema version too
            for _ in 0..5 {
                let op_bytes = next_bytes(&mut state, 1024);
                if let Ok(op) = generate_operation_with_config(&next_sdl, &op_bytes, &op_cfg) {
                    op_pool.push(op);
                }
            }

            // Validate: plan each operation through both fresh and carryover
            let t_plan = Instant::now();
            for op in &op_pool {
                total_ops += 1;
                let fresh_result = planner_fresh.plan(op, None, &opts);
                let carryover_result = planner_carryover.plan(op, None, &opts);

                match (fresh_result, carryover_result) {
                    (Ok(fresh_plan), Ok(carryover_plan)) => {
                        if normalize_plan(&fresh_plan) != normalize_plan(&carryover_plan) {
                            total_mismatches += 1;
                            eprintln!(
                                "HOTSWAP MISMATCH: round={total_rounds} version={version} \
                                 mutation={mutation_kind} cache={max_cache_entries}"
                            );
                            eprintln!("  op: {op}");
                            let fresh_str =
                                serde_json::to_string_pretty(&fresh_plan).unwrap_or_default();
                            let carry_str =
                                serde_json::to_string_pretty(&carryover_plan).unwrap_or_default();
                            for change in similar::TextDiff::from_lines(&fresh_str, &carry_str)
                                .iter_all_changes()
                            {
                                let sign = match change.tag() {
                                    similar::ChangeTag::Delete => "-",
                                    similar::ChangeTag::Insert => "+",
                                    similar::ChangeTag::Equal => continue,
                                };
                                eprint!("  {sign}{change}");
                            }
                            eprintln!();
                        }
                    }
                    (Err(_), Ok(_)) | (Ok(_), Err(_)) => {
                        // One succeeded and the other failed — this is a mismatch
                        total_mismatches += 1;
                        eprintln!(
                            "HOTSWAP ERROR MISMATCH: round={total_rounds} version={version}"
                        );
                    }
                    (Err(_), Err(_)) => {
                        // Both failed — op became invalid after mutation, skip
                        total_skipped += 1;
                    }
                }
            }

            let plan_ms = t_plan.elapsed().as_millis();

            // Track cache growth after planning ops through this version
            let cache_len = planner_carryover.condition_resolver_cache_len();
            if cache_len > max_cache_entries {
                max_cache_entries = cache_len;
            }

            if args.verbose {
                eprintln!(
                    "  [v{version}] plan: {plan_ms}ms ({} ops x2) cache={cache_len}",
                    op_pool.len(),
                );
            }

            // The carryover planner becomes the current planner for the next version
            current_planner = planner_carryover;
            current_sdl = next_sdl;

            if progress_interval > 0 && total_versions % progress_interval == 0 {
                let elapsed = start_time.elapsed();
                eprintln!(
                    "[hotswap] round={total_rounds} version={version} elapsed={:.0}s \
                     ops={total_ops} mismatches={total_mismatches} cache={max_cache_entries}",
                    elapsed.as_secs_f64(),
                );
            }
        }
    }

    let elapsed = start_time.elapsed();
    println!("=== HOT-SWAP VALIDATION REPORT ===");
    println!("elapsed:    {:.1}s", elapsed.as_secs_f64());
    println!("rounds:     {total_rounds}");
    println!("versions:   {total_versions}");
    println!("ops:        tested={total_ops} skipped={total_skipped} mismatches={total_mismatches}");
    println!("max cache:  {max_cache_entries} entries");
    println!(
        "mutations:  add_field={} rename_field={} add_type={} identity={}",
        mutation_counts[0], mutation_counts[1], mutation_counts[2], mutation_counts[3],
    );
    println!();

    if total_mismatches > 0 {
        println!("FAIL: {total_mismatches} hot-swap mismatches found");
        std::process::exit(1);
    } else {
        println!("PASS: all hot-swap plans match fresh plans");
    }
}

// =========================================================================
// Schema mutation helpers
// =========================================================================

/// Mutate a supergraph SDL by adding a new leaf field to the first non-root
/// object type that has a @join__type directive. Returns None if no suitable
/// type can be found.
fn mutate_supergraph_add_field(schema_sdl: &str, version: u64) -> Option<String> {
    // Extract first join__Graph enum value
    let graph_enum_value = {
        let mut in_enum = false;
        let mut found = None;
        for line in schema_sdl.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("enum join__Graph") {
                in_enum = true;
                continue;
            }
            if in_enum {
                if trimmed == "}" {
                    break;
                }
                if trimmed.contains("@join__graph(") {
                    if let Some(name) = trimmed.split_whitespace().next() {
                        found = Some(name.to_string());
                        break;
                    }
                }
            }
        }
        found?
    };

    let mut result = String::with_capacity(schema_sdl.len() + 200);
    let mut inserted = false;
    let mut in_eligible_type = false;
    let mut brace_depth = 0u32;

    for line in schema_sdl.lines() {
        let trimmed = line.trim();

        if !inserted
            && trimmed.starts_with("type ")
            && !trimmed.starts_with("type Query")
            && !trimmed.starts_with("type Mutation")
            && !trimmed.starts_with("type Subscription")
        {
            if trimmed.contains("@join__type") {
                in_eligible_type = true;
                brace_depth = 0;
            }
        }

        if in_eligible_type {
            brace_depth += trimmed.matches('{').count() as u32;
            brace_depth = brace_depth.saturating_sub(trimmed.matches('}').count() as u32);

            if brace_depth == 0 && trimmed.contains('}') && !inserted {
                result.push_str(&format!(
                    "  _hotSwapField{version}: String @join__field(graph: {})\n",
                    graph_enum_value
                ));
                inserted = true;
                in_eligible_type = false;
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    if inserted {
        Some(result)
    } else {
        None
    }
}

/// Mutate by renaming an existing non-key field. This changes the API schema
/// surface without adding or removing condition edges — tests that the cache
/// correctly handles field identity changes.
fn mutate_supergraph_rename_field(schema_sdl: &str, version: u64) -> Option<String> {
    // Find the first field line that has @join__field and doesn't start with underscore
    let mut result = String::with_capacity(schema_sdl.len() + 100);
    let mut renamed = false;

    for line in schema_sdl.lines() {
        let trimmed = line.trim();
        if !renamed
            && trimmed.contains("@join__field")
            && !trimmed.starts_with('_')
            && !trimmed.starts_with("id")
        {
            // Extract field name (first word before the colon)
            if let Some(colon_pos) = trimmed.find(':') {
                let field_name = trimmed[..colon_pos].trim();
                if !field_name.is_empty()
                    && field_name != "id"
                    && !field_name.starts_with("__")
                {
                    let new_name = format!("{field_name}Renamed{version}");
                    let new_line = line.replace(field_name, &new_name);
                    result.push_str(&new_line);
                    result.push('\n');
                    renamed = true;
                    continue;
                }
            }
        }
        result.push_str(line);
        result.push('\n');
    }

    if renamed { Some(result) } else { None }
}

/// Mutate by adding an entirely new type with a query field. This adds new
/// nodes and edges to the QueryGraph — tests that the cache handles graph
/// expansion correctly.
fn mutate_supergraph_add_type(schema_sdl: &str, version: u64) -> Option<String> {
    // Extract first join__Graph enum value
    let graph_enum_value = {
        let mut in_enum = false;
        let mut found = None;
        for line in schema_sdl.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("enum join__Graph") {
                in_enum = true;
                continue;
            }
            if in_enum {
                if trimmed == "}" { break; }
                if trimmed.contains("@join__graph(") {
                    if let Some(name) = trimmed.split_whitespace().next() {
                        found = Some(name.to_string());
                        break;
                    }
                }
            }
        }
        found?
    };

    let type_name = format!("_HotSwapType{version}");
    let field_name = format!("_hotSwapQuery{version}");

    // Add the new type definition and a query field for it
    let mut result = String::with_capacity(schema_sdl.len() + 500);
    let mut added_query_field = false;

    for line in schema_sdl.lines() {
        let trimmed = line.trim();

        // Add a query field inside `type Query`
        if !added_query_field && trimmed == "}" {
            // Check if the previous content was inside type Query
            // We look for a simple heuristic: was the last type declaration Query?
        }

        result.push_str(line);
        result.push('\n');

        // After `type Query ... {` line, inject a new query field
        if !added_query_field && trimmed.starts_with("type Query") && trimmed.contains('{') {
            result.push_str(&format!(
                "  {field_name}: {type_name} @join__field(graph: {graph_enum_value})\n"
            ));
            added_query_field = true;
        }
    }

    if !added_query_field {
        return None;
    }

    // Append the new type definition at the end
    result.push_str(&format!(
        "\ntype {type_name}\n  @join__type(graph: {graph_enum_value})\n{{\n  \
         id: ID!\n  name: String\n}}\n"
    ));

    Some(result)
}
