/// Tests that the shared condition resolver cache produces identical plans
/// regardless of whether the cache was primed by a previous query.
///
/// This is the core correctness proof for Phase 0 of incremental query plan
/// recomputation: hoisting the ConditionResolverCache from per-traversal to
/// per-schema lifetime.

#[test]
fn shared_cache_produces_identical_plans_across_queries() {
    // Schema with @key requiring condition resolution across subgraphs
    let planner = planner!(
        Subgraph1: r#"
          type Query {
            t: T
            s: S
          }

          type T @key(fields: "id") {
            id: ID!
            a: Int
          }

          type S @key(fields: "id") {
            id: ID!
            x: String
          }
        "#,
        Subgraph2: r#"
          type T @key(fields: "id") {
            id: ID!
            a: Int @external
            b: Int @requires(fields: "a")
          }

          type S @key(fields: "id") {
            id: ID!
            x: String @external
            y: String @requires(fields: "x")
          }
        "#,
    );

    // Plan query 1 — this primes the condition resolver cache
    let plan1 = assert_plan!(
        &planner,
        r#"{ t { b } }"#,
        @r###"
        QueryPlan {
          Sequence {
            Fetch(service: "Subgraph1") {
              {
                t {
                  __typename
                  id
                  a
                }
              }
            },
            Flatten(path: "t") {
              Fetch(service: "Subgraph2") {
                {
                  ... on T {
                    __typename
                    id
                    a
                  }
                } =>
                {
                  ... on T {
                    b
                  }
                }
              },
            },
          },
        }
      "###
    );

    // Condition cache should have entries from query 1
    assert!(
        planner.condition_resolver_cache_len() > 0,
        "Expected condition cache to have entries after first query"
    );
    let cache_len_after_q1 = planner.condition_resolver_cache_len();

    // Plan query 2 — uses different type (S) but same cross-subgraph pattern
    // The plan must be correct regardless of cache state from query 1
    assert_plan!(
        &planner,
        r#"{ s { y } }"#,
        @r###"
        QueryPlan {
          Sequence {
            Fetch(service: "Subgraph1") {
              {
                s {
                  __typename
                  id
                  x
                }
              }
            },
            Flatten(path: "s") {
              Fetch(service: "Subgraph2") {
                {
                  ... on S {
                    __typename
                    id
                    x
                  }
                } =>
                {
                  ... on S {
                    y
                  }
                }
              },
            },
          },
        }
      "###
    );

    // Cache should have grown (new condition edges for S)
    assert!(
        planner.condition_resolver_cache_len() > cache_len_after_q1,
        "Expected cache to grow after second query with different type"
    );

    // Re-plan query 1 — must produce byte-identical plan even with cache primed by query 2
    let plan1_again = assert_plan!(
        &planner,
        r#"{ t { b } }"#,
        @r###"
        QueryPlan {
          Sequence {
            Fetch(service: "Subgraph1") {
              {
                t {
                  __typename
                  id
                  a
                }
              }
            },
            Flatten(path: "t") {
              Fetch(service: "Subgraph2") {
                {
                  ... on T {
                    __typename
                    id
                    a
                  }
                } =>
                {
                  ... on T {
                    b
                  }
                }
              },
            },
          },
        }
      "###
    );

    // Plans must be identical
    assert_eq!(
        plan1.to_string(),
        plan1_again.to_string(),
        "Re-planning after cache priming must produce identical plan"
    );

    // Invariant counters must match
    assert_eq!(
        plan1.statistics.evaluated_plan_count.get(),
        plan1_again.statistics.evaluated_plan_count.get(),
        "evaluated_plan_count must be identical"
    );
}

#[test]
fn cache_does_not_corrupt_plans_with_multiple_keys() {
    // Schema where a type has multiple keys — the condition cache must handle
    // ExcludedDestinations correctly across queries
    let planner = planner!(
        Subgraph1: r#"
          type Query {
            t: T
          }

          type T @key(fields: "id") {
            id: ID!
            a: Int
          }
        "#,
        Subgraph2: r#"
          type T @key(fields: "id") @key(fields: "code") {
            id: ID!
            code: String!
            a: Int @external
            b: Int @requires(fields: "a")
          }
        "#,
    );

    // Plan the same query twice — the second time should use cached condition resolutions
    // for the @key edges without corruption from ExcludedDestinations handling
    let plan1 = assert_plan!(
        &planner,
        r#"{ t { b } }"#,
        @r###"
        QueryPlan {
          Sequence {
            Fetch(service: "Subgraph1") {
              {
                t {
                  __typename
                  id
                  a
                }
              }
            },
            Flatten(path: "t") {
              Fetch(service: "Subgraph2") {
                {
                  ... on T {
                    __typename
                    id
                    a
                  }
                } =>
                {
                  ... on T {
                    b
                  }
                }
              },
            },
          },
        }
      "###
    );

    let plan2 = assert_plan!(
        &planner,
        r#"{ t { b } }"#,
        @r###"
        QueryPlan {
          Sequence {
            Fetch(service: "Subgraph1") {
              {
                t {
                  __typename
                  id
                  a
                }
              }
            },
            Flatten(path: "t") {
              Fetch(service: "Subgraph2") {
                {
                  ... on T {
                    __typename
                    id
                    a
                  }
                } =>
                {
                  ... on T {
                    b
                  }
                }
              },
            },
          },
        }
      "###
    );

    assert_eq!(plan1.to_string(), plan2.to_string());
    assert_eq!(
        plan1.statistics.evaluated_plan_count.get(),
        plan2.statistics.evaluated_plan_count.get(),
    );
    // Note: evaluated_plan_paths may differ between runs because the shared
    // condition cache causes sub-traversals to be skipped on cache hits,
    // reducing the path count. This is expected — fewer paths evaluated means
    // the cache is working. The byte-identical plan output and matching
    // evaluated_plan_count are the correct invariants.
    assert!(
        plan2.statistics.evaluated_plan_paths.get() <= plan1.statistics.evaluated_plan_paths.get(),
        "Second run should evaluate at most as many paths as the first (got {} vs {})",
        plan2.statistics.evaluated_plan_paths.get(),
        plan1.statistics.evaluated_plan_paths.get(),
    );
}
