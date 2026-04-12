//! Thread-local wall-clock sub-timers for `compute_nodes_for_tree` in the FDG
//! construction phase.
//!
//! Activated by `QueryPlanningTraversal::compute_best_plan_from_closed_branches`
//! and deactivated after; the final sums are written into `PhaseTimings`.

use std::cell::RefCell;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FdgSubtimings {
    /// Total iterations of the main while-loop in `compute_nodes_for_tree`.
    pub(crate) iterations: u64,
    /// Sum of ns spent in `add_at_path` calls (local_selection_sets + leaf).
    pub(crate) add_at_path_ns: u128,
    /// Sum of ns spent in `compute_nodes_for_key_resolution` calls.
    pub(crate) key_resolution_ns: u128,
    /// Number of key resolution calls.
    pub(crate) key_resolution_calls: u64,
    /// Sum of ns spent in `compute_nodes_for_op_path_element` calls.
    pub(crate) op_path_element_ns: u128,
    /// Number of op path element calls.
    pub(crate) op_path_element_calls: u64,
    /// Sum of ns spent in `compute_nodes_for_root_type_resolution` calls.
    pub(crate) root_type_resolution_ns: u128,
    /// Number of root type resolution calls.
    pub(crate) root_type_resolution_calls: u64,
}

thread_local! {
    static TIMERS: RefCell<Option<FdgSubtimings>> = const { RefCell::new(None) };
}

pub(crate) fn activate() {
    TIMERS.with(|t| {
        *t.borrow_mut() = Some(FdgSubtimings::default());
    });
}

pub(crate) fn deactivate() -> Option<FdgSubtimings> {
    TIMERS.with(|t| t.borrow_mut().take())
}

#[inline]
pub(crate) fn is_active() -> bool {
    TIMERS.with(|t| t.borrow().is_some())
}

#[inline]
pub(crate) fn record_iteration() {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.iterations += 1;
        }
    });
}

#[inline]
pub(crate) fn add_at_path(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.add_at_path_ns += ns;
        }
    });
}

#[inline]
pub(crate) fn add_key_resolution(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.key_resolution_ns += ns;
            s.key_resolution_calls += 1;
        }
    });
}

#[inline]
pub(crate) fn add_op_path_element(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.op_path_element_ns += ns;
            s.op_path_element_calls += 1;
        }
    });
}

#[inline]
pub(crate) fn add_root_type_resolution(ns: u128) {
    TIMERS.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.root_type_resolution_ns += ns;
            s.root_type_resolution_calls += 1;
        }
    });
}
