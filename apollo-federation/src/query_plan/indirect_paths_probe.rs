//! What-if instrumentation probe for `compute_indirect_paths`.
//!
//! Read-only instrumentation that counts, for the duration of a single
//! `QueryPlanningTraversal` open-branches loop, how often a semantic
//! fingerprint passed to `SimultaneousPathsWithLazyIndirectPaths::compute_indirect_paths`
//! matches a fingerprint already seen earlier in the same loop. The counts are
//! written into `QueryPlanningStatistics::phase_timings` at the end of the
//! loop and never influence planning.
//!
//! ## Context
//!
//! The failed T1a experiment (commit 5b48e3f1b) wired up a pointer-identity
//! cache keyed on `Arc<OpGraphPath>` — and measured 0% hit rate, because
//! `SimultaneousPathsWithLazyIndirectPaths::advance_with_operation_element`
//! builds **fresh** `Arc<OpGraphPath>` values via `OpGraphPath::add` before
//! the cartesian product runs. The existing per-instance
//! `lazily_computed_indirect_paths` already catches repeats on the *same*
//! `SimultaneousPathsWithLazyIndirectPaths`; the question this probe answers
//! is whether there is any *cross-instance* reuse at a semantic level within a
//! single traversal. If there isn't, there is no point building a
//! traversal-wide cache at this layer, no matter how clever the key.
//!
//! ## Two fingerprints
//!
//! * **Permissive** — the minimal set of inputs a cache key would need if we
//!   assumed no prefix dependence. Ignores path history entirely. Upper bound
//!   on the hit rate any semantic cache could reach. If this is low, no
//!   layer-local cache can help.
//!
//! * **Strict** — a soundness-preserving superset that includes the prefix
//!   state the Dijkstra in
//!   `advance_with_non_collecting_and_type_preserving_transitions` actually
//!   reads: `last_subgraph_entering_edge_info`, the edge suffix since that
//!   point, `head`, `defer_on_tail`, `runtime_types_of_tail`. Lower bound on
//!   the hit rate a correctness-preserving cache could reach.
//!
//! The gap between permissive and strict tells us how much hit rate is
//! forfeited by insisting on soundness. If permissive is high and strict is
//! low, the sound T1 grafting work (see wiggly-inventing-umbrella.md) is
//! not worth it. If both are comparable, sound T1 is viable.

use std::cell::RefCell;
use std::collections::HashSet;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ProbeCounts {
    /// Total `compute_indirect_paths` invocations observed.
    pub(crate) total: u64,
    /// Invocations whose permissive fingerprint was already in the seen set.
    pub(crate) permissive_repeats: u64,
    /// Invocations whose strict fingerprint was already in the seen set.
    pub(crate) strict_repeats: u64,
}

#[derive(Debug, Default)]
struct IndirectPathsProbe {
    counts: ProbeCounts,
    seen_permissive: HashSet<u64>,
    seen_strict: HashSet<u64>,
}

impl IndirectPathsProbe {
    fn record(&mut self, permissive: u64, strict: u64) {
        self.counts.total += 1;
        if !self.seen_permissive.insert(permissive) {
            self.counts.permissive_repeats += 1;
        }
        if !self.seen_strict.insert(strict) {
            self.counts.strict_repeats += 1;
        }
    }
}

thread_local! {
    static PROBE: RefCell<Option<IndirectPathsProbe>> = const { RefCell::new(None) };
}

/// Activate a fresh probe on the current thread. Any previously-active probe
/// is dropped.
pub(crate) fn activate() {
    PROBE.with(|probe| {
        *probe.borrow_mut() = Some(IndirectPathsProbe::default());
    });
}

/// Deactivate the probe and return the final counts, or `None` if no probe was
/// active.
pub(crate) fn deactivate() -> Option<ProbeCounts> {
    PROBE.with(|probe| probe.borrow_mut().take().map(|p| p.counts))
}

/// Record one `compute_indirect_paths` invocation against the active probe.
/// No-op if no probe is active on the current thread.
pub(crate) fn record(permissive: u64, strict: u64) {
    PROBE.with(|probe| {
        if let Some(probe) = probe.borrow_mut().as_mut() {
            probe.record(permissive, strict);
        }
    });
}

/// Cheap active-check so callers can skip fingerprint computation when the
/// probe is off.
pub(crate) fn is_active() -> bool {
    PROBE.with(|probe| probe.borrow().is_some())
}
