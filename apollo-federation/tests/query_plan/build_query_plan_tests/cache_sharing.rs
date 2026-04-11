use std::time::Instant;

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

/// Quantifies the planning time reduction from the shared condition resolver cache.
///
/// Plans the same query N times through a single planner instance.
/// The first run populates the cache; subsequent runs benefit from cached
/// condition resolutions. Prints wall-clock timing for each run.
///
/// Run with: cargo test -p apollo-federation -- cache_sharing::measure --nocapture
#[test]
fn measure_shared_cache_speedup() {
    let schema =
        std::fs::read_to_string("../examples/graphql/supergraph.graphql").unwrap();
    let supergraph =
        apollo_federation::Supergraph::new(&schema).expect("supergraph should be valid");
    let api_schema = supergraph
        .to_api_schema(apollo_federation::ApiSchemaOptions::default())
        .expect("api schema should be valid");
    let planner = apollo_federation::query_plan::query_planner::QueryPlanner::new(
        &supergraph,
        Default::default(),
    )
    .expect("planner should be created");

    let queries = &[
        ("fetchUser", r#"query fetchUser {
            me {
                id
                name
                username
                reviews {
                    id
                    author { id name }
                }
            }
            recommendedProducts {
                upc weight price shippingEstimate
                reviews { id author { id name } }
            }
            topProducts {
                upc weight price shippingEstimate
                reviews { id author { id name } }
            }
        }"#),
        ("justMe", r#"{ me { id name username } }"#),
        ("topProducts", r#"{ topProducts { upc name price reviews { id body } } }"#),
    ];

    let iterations = 10;
    eprintln!("\n--- Shared condition cache speedup measurement ---");
    eprintln!("  Schema: examples/graphql/supergraph.graphql");
    eprintln!("  Queries: {} distinct, {} iterations each\n", queries.len(), iterations);

    let mut all_timings: Vec<Vec<std::time::Duration>> = Vec::new();

    for (name, query_str) in queries {
        let document = apollo_compiler::ExecutableDocument::parse_and_validate(
            api_schema.schema(),
            *query_str,
            "bench.graphql",
        )
        .expect("query should be valid");

        let mut timings = Vec::new();
        for i in 0..iterations {
            let start = Instant::now();
            let plan = planner
                .build_query_plan(
                    &document,
                    None,
                    apollo_federation::query_plan::query_planner::QueryPlanOptions::default(),
                )
                .expect("plan should succeed");
            let elapsed = start.elapsed();
            timings.push(elapsed);

            if i == 0 {
                eprintln!(
                    "  [{name}] run {}: {:>8.1}µs  (cold — cache len: {}, paths: {}, plans: {})",
                    i + 1,
                    elapsed.as_micros(),
                    planner.condition_resolver_cache_len(),
                    plan.statistics.evaluated_plan_paths.get(),
                    plan.statistics.evaluated_plan_count.get(),
                );
            }
        }

        let cold = timings[0];
        let warm_avg: std::time::Duration =
            timings[1..].iter().sum::<std::time::Duration>() / (timings.len() as u32 - 1);
        let warm_min = *timings[1..].iter().min().unwrap();
        let speedup = cold.as_nanos() as f64 / warm_avg.as_nanos() as f64;

        eprintln!(
            "  [{name}] warm avg: {:>8.1}µs  min: {:>8.1}µs  speedup: {:.2}x",
            warm_avg.as_micros(),
            warm_min.as_micros(),
            speedup,
        );
        all_timings.push(timings);
    }

    // Now measure interleaved planning (different queries sharing cache)
    eprintln!("\n  --- Interleaved (all queries, round-robin, {} rounds) ---", iterations);
    let docs: Vec<_> = queries
        .iter()
        .map(|(_, q)| {
            apollo_compiler::ExecutableDocument::parse_and_validate(
                api_schema.schema(),
                *q,
                "bench.graphql",
            )
            .unwrap()
        })
        .collect();

    // Fresh planner for interleaved test
    let planner2 = apollo_federation::query_plan::query_planner::QueryPlanner::new(
        &supergraph,
        Default::default(),
    )
    .unwrap();

    let mut round_timings: Vec<std::time::Duration> = Vec::new();
    for round in 0..iterations {
        let start = Instant::now();
        for doc in &docs {
            planner2
                .build_query_plan(
                    doc,
                    None,
                    apollo_federation::query_plan::query_planner::QueryPlanOptions::default(),
                )
                .unwrap();
        }
        let elapsed = start.elapsed();
        round_timings.push(elapsed);
        if round == 0 || round == 1 {
            eprintln!(
                "  round {}: {:>8.1}µs  (cache len: {})",
                round + 1,
                elapsed.as_micros(),
                planner2.condition_resolver_cache_len(),
            );
        }
    }

    let round1 = round_timings[0];
    let later_avg: std::time::Duration =
        round_timings[2..].iter().sum::<std::time::Duration>() / (round_timings.len() as u32 - 2);
    let speedup = round1.as_nanos() as f64 / later_avg.as_nanos() as f64;
    eprintln!(
        "  round 3-{} avg: {:>8.1}µs  speedup vs round 1: {:.2}x",
        iterations,
        later_avg.as_micros(),
        speedup,
    );
    eprintln!("  final cache size: {} entries\n", planner2.condition_resolver_cache_len());

    // --- Test multiple exotic supergraphs ---
    let scenarios: &[(&str, &[&str])] = &[
        // 10-subgraph chained @requires — deep condition resolution
        (
            "tests/query_plan/supergraphs/it_handles_longer_require_chain.graphql",
            &[
                r#"{ t { v10 } }"#,
                r#"{ t { v5 } }"#,
                r#"{ t { v1 } }"#,
            ],
        ),
        // 7-subgraph complex requires with fan-out nested objects
        (
            "tests/query_plan/supergraphs/it_handles_complex_require_chain.graphql",
            &[
                r#"{ t { outer } }"#,
                r#"{ t { inner1 inner2 } }"#,
            ],
        ),
        // 4-subgraph interfaces across subgraphs (Book, Magazine implement Product)
        (
            "tests/query_plan/supergraphs/handles_multiple_conditions_on_abstract_types.graphql",
            &[
                r#"{ products { sku dimensions { size weight } reviews { id body } } }"#,
                r#"{ products { id reviews { id body product { id } } } }"#,
            ],
        ),
        // Multiple requires with nested selection sets
        (
            "tests/query_plan/supergraphs/it_handles_multiple_requires_with_multiple_fetches.graphql",
            &[
                r#"{ t { foo } }"#,
                r#"{ t { bar { __typename } } }"#,
                r#"{ t { foo bar { __typename } } }"#,
            ],
        ),
        // Interface object with non-collecting transitions
        (
            "tests/query_plan/supergraphs/test_interface_object_advance_with_non_collecting_and_type_preserving_transitions_ordering.graphql",
            &[
                r#"{ iFromS1 { x } }"#,
            ],
        ),
    ];

    for (schema_path, test_queries) in scenarios {
        let short_name = schema_path
            .rsplit('/')
            .next()
            .unwrap()
            .trim_end_matches(".graphql");
        eprintln!("  --- {short_name} ---");

        let sg_schema = match std::fs::read_to_string(schema_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("    SKIP: {e}");
                continue;
            }
        };
        let sg = match apollo_federation::Supergraph::new(&sg_schema) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("    SKIP (parse): {e}");
                continue;
            }
        };
        let sg_api = sg
            .to_api_schema(apollo_federation::ApiSchemaOptions::default())
            .unwrap();
        let sg_planner = apollo_federation::query_plan::query_planner::QueryPlanner::new(
            &sg,
            Default::default(),
        )
        .unwrap();

        for query_str in *test_queries {
            let doc = match apollo_compiler::ExecutableDocument::parse_and_validate(
                sg_api.schema(),
                *query_str,
                "bench.graphql",
            ) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("    SKIP query `{query_str}`: {e}");
                    continue;
                }
            };

            let mut timings = Vec::new();
            for _ in 0..iterations {
                let start = Instant::now();
                let plan = sg_planner
                    .build_query_plan(
                        &doc,
                        None,
                        apollo_federation::query_plan::query_planner::QueryPlanOptions::default(),
                    )
                    .unwrap();
                let elapsed = start.elapsed();
                timings.push((elapsed, plan.statistics.evaluated_plan_paths.get()));
            }

            let cold = timings[0].0;
            let cold_paths = timings[0].1;
            let warm_avg: std::time::Duration =
                timings[1..].iter().map(|(d, _)| *d).sum::<std::time::Duration>()
                    / (timings.len() as u32 - 1);
            let warm_paths = timings[1].1;
            let speedup = cold.as_nanos() as f64 / warm_avg.as_nanos() as f64;

            let q_short = if query_str.len() > 30 {
                format!("{}...", &query_str[..27])
            } else {
                query_str.to_string()
            };
            eprintln!(
                "    [{:>30}] cold: {:>8}µs ({:>3} paths)  warm: {:>8}µs ({:>3} paths)  {:.2}x",
                q_short,
                cold.as_micros(),
                cold_paths,
                warm_avg.as_micros(),
                warm_paths,
                speedup,
            );
        }
        eprintln!(
            "    cache: {} entries",
            sg_planner.condition_resolver_cache_len()
        );
    }
    eprintln!();
}
