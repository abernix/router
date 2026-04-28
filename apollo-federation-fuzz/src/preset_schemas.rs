//! Deterministic pathological schema constructors for worst-case planning.
//!
//! Unlike the random generator in `subgraph_gen.rs`, these produce specific
//! schema shapes known to stress the query planner's condition resolution,
//! path enumeration, and cross-subgraph fetch chatter.
//!
//! Each preset returns `Vec<SubgraphSdl>` (un-composed subgraphs) and a set
//! of stress queries designed to exercise the pathological patterns.

use std::fmt::Write as _;

use crate::subgraph_gen::SubgraphSdl;

/// Configuration for pathological schema generation.
#[derive(Debug, Clone)]
pub struct PresetConfig {
    /// Number of "extension" subgraphs that all extend the same core entities.
    /// Analogous to the number of teams/departments in a large org.
    /// Realistic range: 10-600.
    pub num_extensions: usize,
}

impl Default for PresetConfig {
    fn default() -> Self {
        Self {
            num_extensions: 50,
        }
    }
}

/// A preset schema with pre-built stress queries.
pub struct PresetSchema {
    pub subgraphs: Vec<SubgraphSdl>,
    pub stress_queries: Vec<PresetQuery>,
}

pub struct PresetQuery {
    pub name: &'static str,
    pub query: String,
}

// ---------------------------------------------------------------------------
// Wide fan-out: N subgraphs all extending the same entity with @requires
// ---------------------------------------------------------------------------

/// Generates a schema where N "department" subgraphs each extend a shared
/// `Product` entity with `@requires(fields: "name price")` and optionally
/// `@requires(fields: "weight")`. A single query touching all N departments
/// forces N condition resolutions through the same entity edges.
///
/// This is the pattern that produces 18-30x warm-cache speedups at N=50
/// and exercises the condition resolver cache most aggressively.
///
/// Schema shape:
/// - `core` subgraph: owns `Product @key(fields: "upc")` with `name`,
///   `price`, `weight` fields. Also owns `Query { products: [Product!]! }`.
/// - `dept{i}` subgraph (× N): extends `Product` with:
///   - `dept{i}Available: Boolean!`
///   - `dept{i}Price: Float!`
///   - `dept{i}Discount: Float! @requires(fields: "name price")`
///   - (even i) `dept{i}ShippingClass: String! @requires(fields: "weight")`
pub fn wide_fan_out(cfg: &PresetConfig) -> PresetSchema {
    let mut subgraphs = Vec::with_capacity(cfg.num_extensions + 1);

    // Core subgraph: owns Product and User
    subgraphs.push(SubgraphSdl::new(
        "core",
        r#"
type Query {
  products(limit: Int = 10): [Product!]!
  product(upc: String!): Product
  me: User
  user(id: ID!): User
}

type Product @key(fields: "upc") {
  upc: String!
  name: String!
  price: Float!
  weight: Float!
  inStock: Boolean!
}

type User @key(fields: "id") @key(fields: "email") {
  id: ID!
  email: String!
  name: String!
  role: String!
}
"#,
    ));

    // Department subgraphs: each extends Product and User with @requires
    for i in 0..cfg.num_extensions {
        let has_weight_requires = i % 2 == 0;
        let has_deep_user_requires = i % 3 == 0;

        let mut sdl = String::new();

        // Product extension
        write!(
            sdl,
            r#"
type Product @key(fields: "upc") {{
  upc: String!
  name: String! @external
  price: Float! @external{weight_external}
  dept{i}Available: Boolean!
  dept{i}Price: Float!
  dept{i}Discount: Float! @requires(fields: "name price"){shipping_class}
}}
"#,
            weight_external = if has_weight_requires {
                "\n  weight: Float! @external"
            } else {
                ""
            },
            shipping_class = if has_weight_requires {
                format!("\n  dept{i}ShippingClass: String! @requires(fields: \"weight\")")
            } else {
                String::new()
            },
        )
        .unwrap();

        // User extension
        if has_deep_user_requires {
            write!(
                sdl,
                r#"
type User @key(fields: "id") {{
  id: ID!
  name: String! @external
  email: String! @external
  dept{i}Preferences: Dept{i}Prefs! @requires(fields: "name email")
  dept{i}Score: Float! @requires(fields: "name")
}}

type Dept{i}Prefs {{
  categories: [String!]!
  notifications: Boolean!
}}
"#
            )
            .unwrap();
        } else {
            write!(
                sdl,
                r#"
type User @key(fields: "id") {{
  id: ID!
  name: String! @external
  dept{i}Preferences: [String!]! @requires(fields: "name")
}}
"#
            )
            .unwrap();
        }

        subgraphs.push(SubgraphSdl::new(format!("dept{i}"), sdl));
    }

    // Build stress queries
    let mut stress_queries = Vec::new();

    // Query 1: touch all dept fields on Product (maximum @requires fan-out)
    let mut q = String::from("{ products { upc name price");
    for i in 0..cfg.num_extensions {
        write!(q, " dept{i}Available dept{i}Price dept{i}Discount").unwrap();
        if i % 2 == 0 {
            write!(q, " dept{i}ShippingClass").unwrap();
        }
    }
    q.push_str(" } }");
    stress_queries.push(PresetQuery {
        name: "product_all_depts",
        query: q,
    });

    // Query 2: touch all dept fields on User
    let mut q = String::from("{ me { id name email");
    for i in 0..cfg.num_extensions {
        if i % 3 == 0 {
            write!(
                q,
                " dept{i}Preferences {{ categories notifications }} dept{i}Score"
            )
            .unwrap();
        } else {
            write!(q, " dept{i}Preferences").unwrap();
        }
    }
    q.push_str(" } }");
    stress_queries.push(PresetQuery {
        name: "user_all_depts",
        query: q,
    });

    // Query 3: half the depts on Product (typical partial query)
    let mut q = String::from("{ product(upc: \"1\") { name price");
    for i in (0..cfg.num_extensions).step_by(2) {
        write!(q, " dept{i}Discount dept{i}Price").unwrap();
    }
    q.push_str(" } }");
    stress_queries.push(PresetQuery {
        name: "product_half_depts",
        query: q,
    });

    // Query 4: single product field — minimal, should be fast
    stress_queries.push(PresetQuery {
        name: "product_simple",
        query: "{ products { upc name price inStock } }".into(),
    });

    PresetSchema {
        subgraphs,
        stress_queries,
    }
}

// ---------------------------------------------------------------------------
// Deep chain: entity references across subgraphs forcing multi-hop traversals
// ---------------------------------------------------------------------------

/// Generates a chain of N subgraphs where each owns a unique entity type
/// and references the next entity in the chain. Querying from the root
/// forces the planner to traverse N hops across N subgraphs.
///
/// Schema shape:
/// - `chain0` subgraph: owns `Query { root: Node0 }` and `Node0 @key(fields: "id")`
///   with `next: Node1` (entity ref to chain1's type).
/// - `chain{i}` subgraph: owns `Node{i} @key(fields: "id")` with:
///   - `value{i}: String!`
///   - `computed{i}: String! @requires(fields: "value{i}")` (self-referencing
///     requires via another subgraph that also extends Node{i})
///   - `next: Node{i+1}` (entity ref to the next chain link)
/// - `crosscut` subgraph: extends ALL Node types with `@external value{i}` and
///   provides a `crosscutField: String!` @requires(fields: "value{i}") on each.
///   This forces the planner to resolve conditions through the crosscut subgraph
///   for every hop in the chain.
pub fn deep_chain(cfg: &PresetConfig) -> PresetSchema {
    let n = cfg.num_extensions;
    let mut subgraphs = Vec::with_capacity(n + 2);

    // Chain subgraphs: each owns one Node type and references the next
    for i in 0..n {
        let mut sdl = String::new();

        if i == 0 {
            write!(
                sdl,
                r#"
type Query {{
  root: Node0
  nodes(limit: Int = 10): [Node0!]!
}}
"#
            )
            .unwrap();
        }

        let next_field = if i + 1 < n {
            format!("\n  next: Node{}", i + 1)
        } else {
            String::new()
        };

        write!(
            sdl,
            r#"
type Node{i} @key(fields: "id") {{
  id: ID!
  value{i}: String!
  label{i}: String!{next_field}
}}
"#
        )
        .unwrap();

        // If referencing the next node, emit a key stub for it
        if i + 1 < n {
            write!(
                sdl,
                r#"
type Node{next} @key(fields: "id") {{
  id: ID!
}}
"#,
                next = i + 1
            )
            .unwrap();
        }

        subgraphs.push(SubgraphSdl::new(format!("chain{i}"), sdl));
    }

    // Crosscut subgraph: extends every Node type with a @requires field
    // that depends on the node's value field (owned by chain{i}).
    // This creates N condition resolution edges through a single subgraph.
    {
        let mut sdl = String::new();
        for i in 0..n {
            write!(
                sdl,
                r#"
type Node{i} @key(fields: "id") {{
  id: ID!
  value{i}: String! @external
  crosscut{i}: String! @requires(fields: "value{i}")
}}
"#
            )
            .unwrap();
        }
        subgraphs.push(SubgraphSdl::new("crosscut", sdl));
    }

    // Stress queries
    let mut stress_queries = Vec::new();

    // Query 1: traverse the full chain with crosscut at each hop
    let depth = n.min(20); // Cap depth for query readability
    let mut q = String::from("{ root ");
    let mut indent = String::new();
    for i in 0..depth {
        write!(q, "{{ id value{i} crosscut{i}").unwrap();
        if i + 1 < depth {
            q.push_str(" next ");
        }
        indent.push_str("} ");
    }
    // Close all braces
    for _ in 0..depth {
        q.push_str(" }");
    }
    q.push_str(" }");
    stress_queries.push(PresetQuery {
        name: "full_chain",
        query: q,
    });

    // Query 2: just the crosscut fields (all condition resolutions, no chaining)
    let mut q = String::from("{ root { id value0 crosscut0");
    if n > 1 {
        write!(q, " next {{ id value1 crosscut1 }}").unwrap();
    }
    q.push_str(" } }");
    stress_queries.push(PresetQuery {
        name: "shallow_crosscut",
        query: q,
    });

    // Query 3: deep chain without crosscut (entity traversal only)
    let depth = n.min(20);
    let mut q = String::from("{ root ");
    for i in 0..depth {
        write!(q, "{{ id value{i} label{i}").unwrap();
        if i + 1 < depth {
            q.push_str(" next ");
        }
    }
    for _ in 0..depth {
        q.push_str(" }");
    }
    q.push_str(" }");
    stress_queries.push(PresetQuery {
        name: "deep_no_crosscut",
        query: q,
    });

    PresetSchema {
        subgraphs,
        stress_queries,
    }
}

// ---------------------------------------------------------------------------
// Mesh: multiple shared entities with cross-subgraph @requires chains
// ---------------------------------------------------------------------------

/// Generates a mesh of `num_extensions` subgraphs that share K core entity
/// types, with each subgraph extending multiple entities and creating
/// @requires dependencies that reference fields owned by other subgraphs.
///
/// This exercises the "back-and-forth chatter" pattern: to resolve one
/// entity's field, the planner must fetch from subgraph A, then B (for
/// @requires), then back to A (for another field that requires data from B).
///
/// Schema shape:
/// - `hub` subgraph: owns 5 core entities (Product, User, Order, Review,
///   Category) with inter-entity references.
/// - `ext{i}` subgraph (× N): extends 2-3 of the core entities with:
///   - Fields that @requires data from the hub
///   - Fields that @requires data from OTHER ext subgraphs
///   - Inter-entity reference fields that create diamond dependencies
pub fn mesh(cfg: &PresetConfig) -> PresetSchema {
    let n = cfg.num_extensions;
    let mut subgraphs = Vec::with_capacity(n + 1);

    // Hub subgraph: owns all core entities
    subgraphs.push(SubgraphSdl::new(
        "hub",
        r#"
type Query {
  products(limit: Int = 10): [Product!]!
  product(id: ID!): Product
  user(id: ID!): User
  order(id: ID!): Order
}

type Product @key(fields: "id") {
  id: ID!
  name: String!
  price: Float!
  weight: Float!
  category: Category!
}

type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
}

type Order @key(fields: "id") {
  id: ID!
  user: User!
  total: Float!
  status: String!
}

type Review @key(fields: "id") {
  id: ID!
  body: String!
  rating: Int!
  author: User!
  product: Product!
}

type Category @key(fields: "id") {
  id: ID!
  name: String!
  parent: Category
}
"#,
    ));

    // Extension subgraphs: each extends a rotating set of core entities
    let core_entities = ["Product", "User", "Order", "Review", "Category"];
    // (field_name, field_type) pairs — types must match hub declarations
    let core_fields: [&[(&str, &str)]; 5] = [
        &[("name", "String!"), ("price", "Float!"), ("weight", "Float!")],
        &[("name", "String!"), ("email", "String!")],
        &[("total", "Float!"), ("status", "String!")],
        &[("body", "String!"), ("rating", "Int!")],
        &[("name", "String!")],
    ];
    let core_keys: [&str; 5] = [
        r#"@key(fields: "id")"#,
        r#"@key(fields: "id")"#,
        r#"@key(fields: "id")"#,
        r#"@key(fields: "id")"#,
        r#"@key(fields: "id")"#,
    ];

    for i in 0..n {
        let mut sdl = String::new();

        // Each extension touches 2-3 entities, rotating through them
        let primary_entity = i % core_entities.len();
        let secondary_entity = (i + 1) % core_entities.len();
        let tertiary_entity = if i % 3 == 0 {
            Some((i + 2) % core_entities.len())
        } else {
            None
        };

        let entities_to_extend: Vec<usize> = {
            let mut v = vec![primary_entity, secondary_entity];
            if let Some(t) = tertiary_entity {
                if !v.contains(&t) {
                    v.push(t);
                }
            }
            v
        };

        // Determine cross-entity linking field (if applicable)
        let cross_link: Option<(usize, &str)> =
            if i % 4 == 0 && entities_to_extend.len() >= 2 {
                Some((entities_to_extend[0], core_entities[entities_to_extend[1]]))
            } else {
                None
            };

        for &e_idx in &entities_to_extend {
            let entity_name = core_entities[e_idx];
            let key = core_keys[e_idx];
            let fields = core_fields[e_idx];

            // Pick 1-2 fields to @requires from
            let (req_field, req_type) = fields[i % fields.len()];
            let req_field2 = if fields.len() > 1 {
                let (f2, t2) = fields[(i + 1) % fields.len()];
                if f2 != req_field { Some((f2, t2)) } else { None }
            } else {
                None
            };

            let mut externals = format!("  {req_field}: {req_type} @external\n");
            let mut requires_sel = req_field.to_string();
            if let Some((f2, t2)) = req_field2 {
                write!(externals, "  {f2}: {t2} @external\n").unwrap();
                write!(requires_sel, " {f2}").unwrap();
            }

            // Add cross-entity reference field if this is the "from" entity
            let link_field = match cross_link {
                Some((from_idx, to_name)) if from_idx == e_idx => {
                    format!("  ext{i}_related_{to_name}: {to_name}\n")
                }
                _ => String::new(),
            };

            write!(
                sdl,
                r#"
type {entity_name} {key} {{
  id: ID!
{externals}  ext{i}_{entity_name}_score: Float! @requires(fields: "{requires_sel}")
  ext{i}_{entity_name}_tag: String!
{link_field}}}
"#
            )
            .unwrap();
        }

        subgraphs.push(SubgraphSdl::new(format!("ext{i}"), sdl));
    }

    // Stress queries
    let mut stress_queries = Vec::new();

    // Query 1: Product with all extension fields
    let mut q = String::from("{ products { id name price weight");
    for i in 0..n {
        if i % core_entities.len() == 0 || (i + 1) % core_entities.len() == 0 {
            write!(q, " ext{i}_Product_score ext{i}_Product_tag").unwrap();
        }
    }
    q.push_str(" } }");
    stress_queries.push(PresetQuery {
        name: "product_all_extensions",
        query: q,
    });

    // Query 2: User with all extension fields
    let mut q = String::from("{ user(id: \"1\") { id name email");
    for i in 0..n {
        if i % core_entities.len() == 1 || (i + 1) % core_entities.len() == 1 {
            write!(q, " ext{i}_User_score ext{i}_User_tag").unwrap();
        }
    }
    q.push_str(" } }");
    stress_queries.push(PresetQuery {
        name: "user_all_extensions",
        query: q,
    });

    // Query 3: Order with nested product + user extensions (chatter)
    let mut q = String::from(
        "{ order(id: \"1\") { id total status user { id name",
    );
    for i in 0..n.min(10) {
        if i % core_entities.len() == 1 || (i + 1) % core_entities.len() == 1 {
            write!(q, " ext{i}_User_score").unwrap();
        }
    }
    q.push_str(" } } }");
    stress_queries.push(PresetQuery {
        name: "order_with_user_extensions",
        query: q,
    });

    // Query 4: minimal baseline
    stress_queries.push(PresetQuery {
        name: "simple_product",
        query: "{ products { id name price } }".into(),
    });

    PresetSchema {
        subgraphs,
        stress_queries,
    }
}

// ---------------------------------------------------------------------------
// Diamond: dense entity graph with multi-key, @provides, @requires chains,
// and diamond-shaped fetch dependencies at 100+ subgraphs
// ---------------------------------------------------------------------------

/// Generates a production-realistic schema topology designed to stress
/// diamond-shaped fetch patterns, @requires chains, @provides, and
/// multi-key entity resolution at scale.
///
/// **Topology:**
/// - 10 "domain" subgraphs, each owning 3 entity types with multiple @key
///   definitions and @provides fields.
/// - N "service" subgraphs, each extending 3-5 entity types from different
///   domains with @requires chains. Service subgraphs @require fields that
///   are @provided by domain subgraphs, AND fields owned by other service
///   subgraphs, creating multi-hop @requires chains.
/// - Diamond pattern: entity E is @key-resolvable from domains D1, D2, D3.
///   Service S1 extends E with @requires from D1. Service S2 extends E
///   with @requires from S1. A query touching both S1 and S2 fields creates
///   a diamond: Query -> D1 -> S1 -> S2, plus Query -> D2 -> S2.
///
/// At num_extensions=100, this produces:
/// - 110 subgraphs (10 domain + 100 service)
/// - 30 entity types with 2-3 @keys each
/// - ~300 @requires fields across service subgraphs
/// - ~30 @provides on domain subgraphs
/// - Diamond fetch patterns with depth 3-4 hops
pub fn diamond(cfg: &PresetConfig) -> PresetSchema {
    let n_services = cfg.num_extensions;
    let n_domains = 10;
    let entities_per_domain = 3;
    let n_entities = n_domains * entities_per_domain; // 30 entity types

    let mut subgraphs = Vec::with_capacity(n_domains + n_services);

    // --- Domain subgraphs (10): each owns 3 entity types ---
    for d in 0..n_domains {
        let mut sdl = String::new();

        // Query root fields (only domain 0 has the root query fields)
        if d == 0 {
            sdl.push_str("type Query {\n");
            for e in 0..n_entities {
                write!(sdl, "  entity{e}(id: ID!): Entity{e}\n").unwrap();
                write!(sdl, "  entity{e}List(limit: Int = 10): [Entity{e}!]!\n").unwrap();
            }
            sdl.push_str("}\n\n");
        }

        let base = d * entities_per_domain;
        for offset in 0..entities_per_domain {
            let e = base + offset;

            // Each entity has 2-3 @key definitions for different resolution paths
            let key2 = if offset % 2 == 0 {
                format!(r#" @key(fields: "sku{e}")"#)
            } else {
                String::new()
            };
            let key3 = if offset == 0 {
                format!(r#" @key(fields: "slug{e}")"#)
            } else {
                String::new()
            };

            write!(
                sdl,
                r#"type Entity{e} @key(fields: "id"){key2}{key3} {{
  id: ID!
  name{e}: String!
  value{e}: Float!
  label{e}: String!
  status{e}: String!"#,
            )
            .unwrap();

            // Second key field (if applicable)
            if offset % 2 == 0 {
                write!(sdl, "\n  sku{e}: String!").unwrap();
            }
            if offset == 0 {
                write!(sdl, "\n  slug{e}: String!").unwrap();
            }

            // Inter-entity references within same domain (local edges, no @provides needed)
            let next_e = base + ((offset + 1) % entities_per_domain);
            write!(sdl, "\n  related{e}: Entity{next_e}").unwrap();

            // Cross-domain entity reference (creates cross-subgraph edges,
            // forces entity resolution through a different domain)
            let cross_domain = (d + 1) % n_domains;
            let cross_e = cross_domain * entities_per_domain;
            write!(sdl, "\n  linkedTo{e}: Entity{cross_e}").unwrap();

            sdl.push_str("\n}\n\n");
        }

        // Stub key-only types for entities from other domains that we reference
        let mut stubbed = std::collections::HashSet::new();

        // Cross-domain link target
        let cross_domain = (d + 1) % n_domains;
        let cross_e = cross_domain * entities_per_domain;
        if cross_domain != d && stubbed.insert(cross_e) {
            write!(
                sdl,
                r#"type Entity{cross_e} @key(fields: "id") {{
  id: ID!
}}

"#
            )
            .unwrap();
        }

        // Domain 0 has Query root fields referencing ALL entity types
        if d == 0 {
            for other_d in 1..n_domains {
                let other_base = other_d * entities_per_domain;
                for other_offset in 0..entities_per_domain {
                    let other_e = other_base + other_offset;
                    if stubbed.insert(other_e) {
                        write!(
                            sdl,
                            r#"type Entity{other_e} @key(fields: "id") {{
  id: ID!
}}

"#
                        )
                        .unwrap();
                    }
                }
            }
        }

        subgraphs.push(SubgraphSdl::new(format!("domain{d}"), sdl));
    }

    // --- Service subgraphs (N): each extends 3-5 entities with @requires ---
    for s in 0..n_services {
        let mut sdl = String::new();

        // Each service touches 3-5 entities, rotating through all 30
        let n_entities_touched = 3 + (s % 3); // 3, 4, or 5
        let mut entities_for_service: Vec<usize> = Vec::new();
        for j in 0..n_entities_touched {
            let e = (s * 3 + j * 7) % n_entities; // spread across entities
            if !entities_for_service.contains(&e) {
                entities_for_service.push(e);
            }
        }
        // Ensure at least 3 unique entities
        while entities_for_service.len() < 3 {
            let e = (s + entities_for_service.len() * 11) % n_entities;
            if !entities_for_service.contains(&e) {
                entities_for_service.push(e);
            }
        }

        for (j, &e) in entities_for_service.iter().enumerate() {
            // Determine what this service @requires:
            // - Always requires the base fields (name, value) from the domain
            // - Every 3rd service ALSO requires a field from another service subgraph,
            //   creating multi-hop @requires chains
            let requires_base = format!("name{e} value{e}");

            // Cross-service @requires: svc_i requires svc_{i-1}'s field on the same entity,
            // but ONLY if svc_{i-1} actually extends this entity.
            let cross_svc_req = if s > 0 && j == 0 {
                let prev_svc = s - 1;
                let prev_entities = entities_for_service_fn(prev_svc, n_entities);
                if prev_entities.contains(&e) {
                    Some(format!("svc{prev_svc}_tag{e}"))
                } else {
                    None
                }
            } else {
                None
            };

            let full_requires = match &cross_svc_req {
                Some(cross_field) => format!("{requires_base} {cross_field}"),
                None => requires_base.clone(),
            };

            // Build entity extension
            write!(
                sdl,
                r#"type Entity{e} @key(fields: "id") {{
  id: ID!
  name{e}: String! @external
  value{e}: Float! @external"#,
            )
            .unwrap();

            // External declaration for cross-service field
            if let Some(cross_field) = &cross_svc_req {
                write!(sdl, "\n  {cross_field}: String! @external").unwrap();
            }

            // Own fields with @requires
            write!(
                sdl,
                r#"
  svc{s}_score{e}: Float! @requires(fields: "{full_requires}")
  svc{s}_tag{e}: String!
  svc{s}_computed{e}: Float! @requires(fields: "value{e}")"#,
            )
            .unwrap();

            // Diamond-creating field: every 4th service adds a field that
            // @requires from TWO different external sources, forcing the planner
            // to build a diamond fetch pattern
            if s % 4 == 0 && j < entities_for_service.len() - 1 {
                let other_e = entities_for_service[j + 1];
                write!(
                    sdl,
                    r#"
  svc{s}_diamond{e}: String! @requires(fields: "name{e} label{e}")"#,
                )
                .unwrap();
                // Also add label as external
                write!(sdl, "\n  label{e}: String! @external").unwrap();

                // Cross-entity reference creating diamond fetches
                write!(sdl, "\n  svc{s}_linked{e}: Entity{other_e}").unwrap();
            }

            sdl.push_str("\n}\n\n");
        }

        // Stub types for entities we reference but don't extend
        let mut stubbed = std::collections::HashSet::new();
        for (j, &e) in entities_for_service.iter().enumerate() {
            if s % 4 == 0 && j < entities_for_service.len() - 1 {
                let other_e = entities_for_service[j + 1];
                if !entities_for_service.contains(&other_e) && stubbed.insert(other_e) {
                    write!(
                        sdl,
                        r#"type Entity{other_e} @key(fields: "id") {{
  id: ID!
}}

"#
                    )
                    .unwrap();
                }
            }
        }

        subgraphs.push(SubgraphSdl::new(format!("svc{s}"), sdl));
    }

    // --- Stress queries ---
    let mut stress_queries = Vec::new();

    // Query 1: Touch ALL service fields on Entity0 (maximum diamond depth)
    {
        let mut q = String::from("{ entity0(id: \"1\") { id name0 value0 label0 status0");
        for s in 0..n_services {
            // Check if service s extends Entity0
            let entities_for_s = entities_for_service_fn(s, n_entities);
            if entities_for_s.contains(&0) {
                write!(q, " svc{s}_score0 svc{s}_tag0 svc{s}_computed0").unwrap();
                if s % 4 == 0 {
                    write!(q, " svc{s}_diamond0").unwrap();
                }
            }
        }
        q.push_str(" } }");
        stress_queries.push(PresetQuery {
            name: "entity0_all_services",
            query: q,
        });
    }

    // Query 2: Cross-entity traversal through linked fields
    {
        let q = format!(
            "{{ entity0(id: \"1\") {{ id name0 related0 {{ id name1 related1 {{ id name2 }} }} \
             linkedTo0 {{ id name{linked} }} }} }}",
            linked = entities_per_domain,
        );
        stress_queries.push(PresetQuery {
            name: "cross_entity_traversal",
            query: q,
        });
    }

    // Query 3: Multiple entities with service extensions (wide + deep)
    {
        let mut q = String::from("{ ");
        for e in 0..5.min(n_entities) {
            write!(q, "e{e}: entity{e}(id: \"{e}\") {{ id name{e} value{e}").unwrap();
            // First 3 services that extend this entity
            let mut svc_count = 0;
            for s in 0..n_services {
                if svc_count >= 3 { break; }
                let es = entities_for_service_fn(s, n_entities);
                if es.contains(&e) {
                    write!(q, " svc{s}_score{e} svc{s}_tag{e}").unwrap();
                    svc_count += 1;
                }
            }
            q.push_str(" } ");
        }
        q.push('}');
        stress_queries.push(PresetQuery {
            name: "multi_entity_wide",
            query: q,
        });
    }

    // Query 4: minimal baseline
    stress_queries.push(PresetQuery {
        name: "simple_entity0",
        query: "{ entity0(id: \"1\") { id name0 value0 } }".into(),
    });

    PresetSchema {
        subgraphs,
        stress_queries,
    }
}

/// Helper: compute which entities service `s` extends (deterministic, matches diamond() logic)
fn entities_for_service_fn(s: usize, n_entities: usize) -> Vec<usize> {
    let n_touched = 3 + (s % 3);
    let mut es = Vec::new();
    for j in 0..n_touched {
        let e = (s * 3 + j * 7) % n_entities;
        if !es.contains(&e) { es.push(e); }
    }
    while es.len() < 3 {
        let e = (s + es.len() * 11) % n_entities;
        if !es.contains(&e) { es.push(e); }
    }
    es
}
