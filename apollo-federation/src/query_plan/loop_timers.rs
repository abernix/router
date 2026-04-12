//! Thread-local wall-clock sub-timers for the open-branches while-loop.
//!
//! Activated by `QueryPlanningTraversal::find_best_plan_inner` before the loop
//! and deactivated after; the final sums are written into `PhaseTimings`.
//!
//! Motivation: the Phase 3a ceiling probe (commit 5b177971c) proved that
//! `compute_indirect_paths` is *not* the open-branches loop hotspot — at most
//! ~66 calls account for a 20ms warm loop on product_stress at 65 subgraphs.
//! That means the time lives in one of the other sub-calls made by
//! `SimultaneousPathsWithLazyIndirectPaths::advance_with_operation_element`:
//!
//! * `OpGraphPath::advance_with_operation_element` — per-path direct advance
//! * `SimultaneousPathsWithLazyIndirectPaths::indirect_options` — the lazy
//!   cache wrapper that may call `compute_indirect_paths` underneath
//! * the indirect-path `OpGraphPath::advance_with_operation_element` calls
//!   after indirect-options returns a set of non-collecting paths
//! * `SimultaneousPaths::flat_cartesian_product` — fan-out over all paths'
//!   options
//! * (implicit) everything else in the outer advance: context updates,
//!   post-filter bookkeeping, `create_lazy_options`
//!
//! Each sub-call is wrapped in `Instant::now() / elapsed()` only when the
//! timers are active, so production planning pays nothing.
//!
//! The recorded totals are *sums across the whole traversal*, not per-call
//! averages. Per-call rates can be reconstructed by dividing by
//! `evaluated_plan_paths` or the probe's `total` count.

use std::cell::RefCell;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct LoopSubtimings {
    /// Sum of wall-clock ns spent inside the outer
    /// `SimultaneousPathsWithLazyIndirectPaths::advance_with_operation_element`
    /// calls from `handle_open_branch`. Everything below is a subset of this.
    pub(crate) outer_advance_ns: u128,
    /// Sum of ns spent in `OpGraphPath::advance_with_operation_element` for
    /// the *direct* (non-indirect) advance path.
    pub(crate) direct_advance_ns: u128,
    /// Sum of ns spent in
    /// `SimultaneousPathsWithLazyIndirectPaths::indirect_options`, including
    /// any underlying `compute_indirect_paths` work.
    pub(crate) indirect_options_ns: u128,
    /// Sum of ns spent in the per-non-collecting-path
    /// `OpGraphPath::advance_with_operation_element` calls made *after*
    /// `indirect_options` returns.
    pub(crate) indirect_advance_ns: u128,
    /// Sum of ns spent in `SimultaneousPaths::flat_cartesian_product`.
    pub(crate) cartesian_product_ns: u128,
}

thread_local! {
    static TIMERS: RefCell<Option<LoopSubtimings>> = const { RefCell::new(None) };
}

/// Activate a fresh set of sub-timers on the current thread.
pub(crate) fn activate() {
    TIMERS.with(|t| {
        *t.borrow_mut() = Some(LoopSubtimings::default());
    });
}

/// Deactivate and return the accumulated sums, or `None` if no timers were
/// active.
pub(crate) fn deactivate() -> Option<LoopSubtimings> {
    TIMERS.with(|t| t.borrow_mut().take())
}

/// Cheap active-check so hot-path callers can skip `Instant::now()` when the
/// timers are off.
#[inline]
pub(crate) fn is_active() -> bool {
    TIMERS.with(|t| t.borrow().is_some())
}

#[inline]
pub(crate) fn add_outer_advance(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.outer_advance_ns += ns;
        }
    });
}

#[inline]
pub(crate) fn add_direct_advance(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.direct_advance_ns += ns;
        }
    });
}

#[inline]
pub(crate) fn add_indirect_options(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.indirect_options_ns += ns;
        }
    });
}

#[inline]
pub(crate) fn add_indirect_advance(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.indirect_advance_ns += ns;
        }
    });
}

#[inline]
pub(crate) fn add_cartesian_product(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.cartesian_product_ns += ns;
        }
    });
}
