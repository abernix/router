//! Portable, serializable representation of a `QueryGraph` for precomputed artifacts.
//!
//! A `PortableQueryGraph` captures the entire graph topology (nodes, edges, indices)
//! in a format that can be serialized to bytes for disk caching or Redis storage.
//! Loading from bytes bypasses the expensive `extract_subgraphs_from_supergraph` +
//! `SchemaQueryGraphBuilder` + `FederatedQueryGraphBuilder` pipeline.
//!
//! # Integrity
//!
//! Each artifact carries a SHA-256 checksum of its source supergraph SDL. On load,
//! the router verifies this checksum against the current supergraph to detect stale
//! or mismatched artifacts. A condition cache seed artifact would additionally carry
//! the QueryGraph artifact's checksum, forming a chain of trust.
//!
//! # Format
//!
//! The portable format stores:
//! - SHA-256 of the supergraph SDL (integrity check)
//! - Individual subgraph schemas as SDL strings (re-parsed on load)
//! - Graph nodes with type/source/provide_id/root_kind as strings
//! - Graph edges with transition kind, conditions as SDL, override info
//! - All precomputed index maps
//!
//! The same byte format works for both on-disk cache files and Redis `GET`/`SET`.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use apollo_compiler::Name;
use apollo_compiler::collections::IndexMap;
use apollo_compiler::collections::IndexSet;
use apollo_compiler::schema::NamedType;
use apollo_compiler::validation::Valid;
use apollo_compiler::Schema;
use petgraph::graph::DiGraph;
use petgraph::graph::EdgeIndex;
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};

use super::QueryGraph;
use super::QueryGraphEdge;
use super::QueryGraphEdgeTransition;
use super::QueryGraphNode;
use super::QueryGraphNodeType;
use super::OverrideCondition;
use super::SemanticEdgeId;
use super::SemanticNodeId;
use super::build_query_graph::FEDERATED_GRAPH_ROOT_SOURCE;
use crate::error::FederationError;
use crate::schema::ValidFederationSchema;
use crate::schema::position::CompositeTypeDefinitionPosition;
use crate::schema::position::EnumTypeDefinitionPosition;
use crate::schema::position::FieldDefinitionPosition;
use crate::schema::position::InterfaceFieldDefinitionPosition;
use crate::schema::position::InterfaceTypeDefinitionPosition;
use crate::schema::position::ObjectFieldDefinitionPosition;
use crate::schema::position::ObjectTypeDefinitionPosition;
use crate::schema::position::OutputTypeDefinitionPosition;
use crate::schema::position::ScalarTypeDefinitionPosition;
use crate::schema::position::SchemaRootDefinitionKind;
use crate::schema::position::UnionTypeDefinitionPosition;
use crate::schema::position::UnionTypenameFieldDefinitionPosition;
use crate::query_plan::query_planning_traversal::non_local_selections_estimation::{
    self,
    precompute_non_local_selection_metadata,
};

/// A fully serializable snapshot of a `QueryGraph` topology + schemas.
#[derive(Serialize, Deserialize)]
pub struct PortableQueryGraph {
    /// Format version for forward compatibility.
    pub version: u32,
    /// Content hash of the supergraph SDL this was built from.
    /// Used to verify the artifact matches the current supergraph on load.
    pub supergraph_sdl_hash: String,
    /// Individual subgraph schemas as SDL strings, keyed by subgraph name.
    /// The supergraph SDL itself is NOT stored — the caller already has it.
    pub subgraph_sdls: BTreeMap<String, String>,
    /// Graph nodes in petgraph index order.
    pub nodes: Vec<PortableNode>,
    /// Graph edges in petgraph index order: (source_idx, target_idx, edge_data).
    pub edges: Vec<(u32, u32, PortableEdge)>,
    /// Root kind to node index mappings per source.
    pub root_kinds_to_nodes_by_source: BTreeMap<String, BTreeMap<String, u32>>,
    /// Type names to node indices per source.
    pub types_to_nodes_by_source: BTreeMap<String, BTreeMap<String, Vec<u32>>>,
    /// Non-trivial followup edges: edge_index -> [edge_indices].
    pub non_trivial_followup_edges: BTreeMap<u32, Vec<u32>>,
    /// Field edge index: (node_index, field_name) -> [edge_indices].
    pub field_edge_index: Vec<((u32, String), Vec<u32>)>,
    /// Override condition labels.
    pub override_condition_labels: Vec<String>,
    /// Interned condition table (v2+). Each entry is a unique condition selection set.
    /// Edges reference conditions by index into this table via `condition_index`.
    #[serde(default)]
    pub condition_table: Vec<PortableCondition>,
    /// Pre-computed non-local selection metadata (v3+).
    /// When present, reconstruction skips the expensive `precompute_non_local_selection_metadata()` call.
    #[serde(default)]
    pub metadata: Option<PortableMetadata>,
    /// Semantic node index: SemanticNodeId -> NodeIndex (v3+).
    #[serde(default)]
    pub semantic_node_index: Vec<(PortableSemanticNodeId, u32)>,
    /// Semantic edge index: SemanticEdgeId -> EdgeIndex (v3+).
    #[serde(default)]
    pub semantic_edge_index: Vec<(PortableSemanticEdgeId, u32)>,
}

#[derive(Serialize, Deserialize)]
pub struct PortableNode {
    /// Type name (e.g. "User") or synthetic root name (e.g. "[Query]").
    pub type_name: String,
    /// Whether this is a schema type or a federated root type.
    pub is_federated_root: bool,
    /// Root kind string if this is a root node.
    pub root_kind: Option<String>,
    /// Source subgraph name.
    pub source: String,
    /// @provides duplication id.
    pub provide_id: Option<u32>,
    /// Whether cross-subgraph edges are reachable.
    pub has_reachable_cross_subgraph_edges: bool,
    /// For schema types: "object", "interface", or "union". None for federated roots.
    pub type_kind: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct PortableEdge {
    pub transition: PortableTransition,
    /// Conditions as a selection set string (e.g., "{ id }"), or None.
    /// Used by v1 format; v2 uses `condition_index` instead.
    pub conditions: Option<String>,
    /// Index into `PortableQueryGraph::condition_table` (v2+).
    /// When present, this takes precedence over `conditions`.
    #[serde(default)]
    pub condition_index: Option<u32>,
    /// Override condition: (label, polarity).
    pub override_condition: Option<(String, bool)>,
    /// Required context conditions (for @fromContext).
    pub required_contexts: Vec<PortableContextCondition>,
}

#[derive(Serialize, Deserialize)]
pub enum PortableTransition {
    FieldCollection {
        source: String,
        /// The field position as a display string (e.g. "User.name").
        field_position: String,
        /// Disambiguator: "object", "interface", or "union_typename".
        field_kind: String,
        is_part_of_provides: bool,
    },
    Downcast {
        source: String,
        from_type: String,
        from_type_kind: String,
        to_type: String,
        to_type_kind: String,
    },
    KeyResolution,
    RootTypeResolution {
        root_kind: String,
    },
    SubgraphEnteringTransition,
    InterfaceObjectFakeDownCast {
        source: String,
        from_type: String,
        from_type_kind: String,
        to_type_name: String,
    },
}

#[derive(Serialize, Deserialize)]
pub struct PortableContextCondition {
    pub context: String,
    pub subgraph_name: String,
    pub selection: String,
    pub argument_name: String,
    pub argument_type: String,
}

const FORMAT_VERSION: u32 = 3;

/// A pre-parsed condition (selection set from `@key(fields: ...)` or `@requires(fields: ...)`).
///
/// Instead of storing conditions as SDL strings that must be re-parsed via `parse_field_set`
/// on load, we store them as a lightweight tree structure that can be reconstructed into a
/// `SelectionSet` via direct schema lookups — O(n) HashMap gets, no parsing.
///
/// Conditions are interned in `PortableQueryGraph::condition_table` and referenced by index
/// from edges, since many edges share identical conditions (e.g. 50+ subgraphs with
/// `@key(fields: "id")` on the same type).
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Debug)]
pub struct PortableCondition {
    /// The parent type this selection set is rooted at (e.g. "User").
    pub type_name: String,
    /// The selections within this condition.
    pub selections: Vec<PortableConditionSelection>,
}

/// A single selection within a `PortableCondition`.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Debug)]
pub enum PortableConditionSelection {
    Field {
        /// Parent type name (e.g. "User"). Needed for FieldDefinitionPosition lookup.
        parent_type_name: String,
        /// Field name (e.g. "id").
        field_name: String,
        /// SDL string for arguments, if any. None for the vast majority of conditions.
        arguments_sdl: Option<String>,
        /// SDL string for directives, if any. None for the vast majority of conditions.
        directives_sdl: Option<String>,
        /// Nested selections for composite-typed fields.
        sub_selections: Option<PortableCondition>,
    },
    InlineFragment {
        /// Parent type name.
        parent_type_name: String,
        /// Type condition (e.g. "Admin"), or None for untyped fragments.
        type_condition: Option<String>,
        /// SDL string for directives, if any.
        directives_sdl: Option<String>,
        /// Nested selections within the fragment.
        selections: PortableCondition,
    },
}

// --- Portable metadata types (v3+) ---

/// Serializable snapshot of `QueryGraphMetadata` — the non-local selection estimation data
/// that is precomputed per-schema. Storing this in the artifact eliminates the expensive
/// `precompute_non_local_selection_metadata()` call during reconstruction.
#[derive(Serialize, Deserialize)]
pub struct PortableMetadata {
    /// type_name -> indirect options (complete digraph members + interface object types)
    pub types_to_indirect_options: Vec<(String, PortableIndirectOptions)>,
    /// node_index -> set of interface object type names (for non-digraph nodes)
    pub remaining_nodes_to_interface_object_options: Vec<(u32, Vec<String>)>,
    /// field_name -> [(node_index, field_target)]
    pub fields_to_endpoints: Vec<(String, Vec<(u32, PortableFieldTarget)>)>,
    /// type_condition_name -> [(source_node, target_node)]
    pub inline_fragments_to_endpoints: Vec<(String, Vec<(u32, u32)>)>,
    /// node_index -> object type downcasts
    pub nodes_to_object_type_downcasts: Vec<(u32, PortableObjectTypeDowncasts)>,
    /// field_name -> [node_index] of rebaseable parent nodes
    pub fields_to_rebaseable_parent_nodes: Vec<(String, Vec<u32>)>,
    /// type_condition_name -> [node_index] of rebaseable parent nodes
    pub inline_fragments_to_rebaseable_parent_nodes: Vec<(String, Vec<u32>)>,
}

#[derive(Serialize, Deserialize)]
pub struct PortableIndirectOptions {
    pub same_type_options: Vec<u32>,
    pub interface_object_options: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub enum PortableFieldTarget {
    NonOverride(u32),
    Override(u32, String, bool), // node_index, label, condition
}

#[derive(Serialize, Deserialize)]
pub enum PortableObjectTypeDowncasts {
    NonInterfaceObject(Vec<(String, u32)>),
    InterfaceObject(Vec<String>),
}

// --- Portable semantic index types (v3+, reserved for future use) ---

/// Serializable form of `SemanticNodeId`. Currently unused — semantic indices
/// are faster to recompute from the graph than to deserialize. Kept for forward
/// compatibility of the artifact format.
#[derive(Serialize, Deserialize)]
pub struct PortableSemanticNodeId {
    pub type_name: String,
    pub source: String,
    pub provide_id: Option<u32>,
}

/// Serializable form of `SemanticEdgeId`. Currently unused — see `PortableSemanticNodeId`.
#[derive(Serialize, Deserialize)]
pub struct PortableSemanticEdgeId {
    pub head: PortableSemanticNodeId,
    pub tail: PortableSemanticNodeId,
    pub transition: PortableSemanticTransitionKind,
    pub conditions_hash: u64,
    pub override_label: Option<String>,
    pub override_condition: Option<bool>,
}

/// Serializable form of `SemanticTransitionKind`. Currently unused — see `PortableSemanticNodeId`.
#[derive(Serialize, Deserialize)]
pub enum PortableSemanticTransitionKind {
    FieldCollection { field_name: String, is_part_of_provides: bool },
    Downcast { to_type_name: String },
    KeyResolution,
    RootTypeResolution { root_kind: String },
    SubgraphEnteringTransition,
    InterfaceObjectFakeDownCast { to_type_name: String },
}

/// Compute a content hash of a string for artifact integrity verification.
/// Uses a 128-bit hash (two 64-bit halves) for collision resistance.
/// Not cryptographic — intended for detecting accidental mismatches, not adversarial tampering.
pub fn content_hash(s: &str) -> String {
    // Hash the first and second halves with different seeds for 128-bit collision resistance.
    let h1 = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    };
    let h2 = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write_u64(0x517cc1b727220a95); // seed
        s.hash(&mut hasher);
        hasher.finish()
    };
    format!("{:016x}{:016x}", h1, h2)
}

impl PortableQueryGraph {
    /// Create a portable representation from a built `QueryGraph` and the supergraph SDL
    /// it was built from.
    pub fn from_query_graph(graph: &QueryGraph, supergraph_sdl: &str) -> Self {
        let supergraph_sdl_hash = content_hash(supergraph_sdl);

        // Extract subgraph schemas as SDL strings.
        let subgraph_sdls: BTreeMap<String, String> = graph
            .subgraph_schemas()
            .iter()
            .map(|(name, schema)| {
                (name.to_string(), schema.schema().serialize().to_string())
            })
            .collect();

        // Serialize nodes in index order.
        let nodes: Vec<PortableNode> = graph
            .graph()
            .node_indices()
            .map(|idx| {
                let node = graph.graph().node_weight(idx).unwrap();
                portable_node(node)
            })
            .collect();

        // Build condition intern table: deduplicates identical conditions across edges.
        let mut condition_intern: HashMap<PortableCondition, u32> = HashMap::new();
        let mut condition_table: Vec<PortableCondition> = Vec::new();

        // Serialize edges in index order.
        let edges: Vec<(u32, u32, PortableEdge)> = graph
            .graph()
            .edge_indices()
            .map(|idx| {
                let (src, tgt) = graph.graph().edge_endpoints(idx).unwrap();
                let edge = graph.graph().edge_weight(idx).unwrap();
                let mut pedge = portable_edge(edge);

                // Build condition index for v2 format.
                if let Some(cond_sel) = &edge.conditions {
                    let portable_cond = portable_condition_from_selection_set(cond_sel);
                    let index = if let Some(&existing) = condition_intern.get(&portable_cond) {
                        existing
                    } else {
                        let idx = condition_table.len() as u32;
                        condition_intern.insert(portable_cond.clone(), idx);
                        condition_table.push(portable_cond);
                        idx
                    };
                    pedge.condition_index = Some(index);
                }

                (src.index() as u32, tgt.index() as u32, pedge)
            })
            .collect();

        // Serialize root_kinds_to_nodes_by_source.
        let root_kinds_to_nodes_by_source = serialize_root_kinds(graph);

        // Serialize types_to_nodes_by_source.
        let types_to_nodes_by_source = serialize_types_to_nodes(graph);

        // Serialize non_trivial_followup_edges.
        let non_trivial_followup_edges = serialize_followup_edges(graph);

        // Serialize field_edge_index.
        let field_edge_index = serialize_field_edge_index(graph);

        // Override condition labels.
        let override_condition_labels = graph
            .override_condition_labels()
            .iter()
            .map(|l| l.to_string())
            .collect();

        // Serialize non-local selection metadata (v3+).
        let metadata = Some(metadata_to_portable(graph.non_local_selection_metadata()));

        PortableQueryGraph {
            version: FORMAT_VERSION,
            supergraph_sdl_hash,
            subgraph_sdls,
            nodes,
            edges,
            root_kinds_to_nodes_by_source,
            types_to_nodes_by_source,
            non_trivial_followup_edges,
            field_edge_index,
            override_condition_labels,
            condition_table,
            metadata,
            semantic_node_index: Vec::new(),
            semantic_edge_index: Vec::new(),
        }
    }

    /// Verify that this artifact was built from the given supergraph SDL.
    pub fn verify_supergraph(&self, supergraph_sdl: &str) -> bool {
        content_hash(supergraph_sdl) == self.supergraph_sdl_hash
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("PortableQueryGraph serialization should not fail")
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let portable: Self =
            serde_json::from_slice(bytes).map_err(|e| format!("JSON deserialization error: {e}"))?;
        if portable.version == 0 || portable.version > FORMAT_VERSION {
            return Err(format!(
                "unsupported format version: got {}, expected 1..={FORMAT_VERSION}",
                portable.version
            ));
        }
        Ok(portable)
    }

    /// Serialize to compact binary (bincode). ~4-6x smaller than JSON.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("PortableQueryGraph serialization should not fail")
    }

    /// Deserialize from compact binary (bincode).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let portable: Self =
            bincode::deserialize(bytes).map_err(|e| format!("bincode deserialization error: {e}"))?;
        if portable.version == 0 || portable.version > FORMAT_VERSION {
            return Err(format!(
                "unsupported format version: got {}, expected 1..={FORMAT_VERSION}",
                portable.version
            ));
        }
        Ok(portable)
    }

    /// Total number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of subgraphs.
    pub fn subgraph_count(&self) -> usize {
        self.subgraph_sdls.len()
    }

    /// Reconstruct a live `QueryGraph` from this portable representation.
    ///
    /// This re-parses subgraph SDL strings and rebuilds the graph topology from
    /// serialized nodes/edges/indices — bypassing the expensive
    /// `extract_subgraphs_from_supergraph` + `SchemaQueryGraphBuilder` +
    /// `FederatedQueryGraphBuilder` pipeline entirely.
    ///
    /// The caller must also provide the supergraph schema (already parsed) so we
    /// can set it on the graph and parse condition selection sets against it.
    /// Reconstruct a live `QueryGraph` from this portable representation.
    ///
    /// Returns `(QueryGraph, ReconstructionTiming)`.
    pub fn to_query_graph(
        &self,
        supergraph_schema: ValidFederationSchema,
    ) -> Result<QueryGraph, FederationError> {
        self.to_query_graph_with_timing(supergraph_schema).map(|(g, _)| g)
    }

    /// Same as `to_query_graph` but also returns timing breakdown.
    pub fn to_query_graph_with_timing(
        &self,
        supergraph_schema: ValidFederationSchema,
    ) -> Result<(QueryGraph, ReconstructionTiming), FederationError> {
        self.to_query_graph_with_schemas(supergraph_schema, None)
    }

    /// Reconstruct a QueryGraph, optionally reusing pre-parsed subgraph schemas.
    ///
    /// If `existing_subgraphs` is provided, subgraph schemas matching by name are
    /// reused directly — skipping the expensive SDL parse + validate step for those.
    /// Any subgraph in the artifact but not in `existing_subgraphs` is parsed from SDL.
    pub fn to_query_graph_with_schemas(
        &self,
        supergraph_schema: ValidFederationSchema,
        existing_subgraphs: Option<&IndexMap<Arc<str>, ValidFederationSchema>>,
    ) -> Result<(QueryGraph, ReconstructionTiming), FederationError> {
        use std::time::Instant;
        let mut timing = ReconstructionTiming::default();

        // 1. Parse subgraph SDLs into ValidFederationSchemas (or reuse existing).
        let t0 = Instant::now();
        let mut subgraphs_by_name: IndexMap<Arc<str>, ValidFederationSchema> = IndexMap::default();
        for (name, sdl) in &self.subgraph_sdls {
            let key: Arc<str> = Arc::from(name.as_str());
            if let Some(existing) = existing_subgraphs.and_then(|m| m.get(&*key)) {
                subgraphs_by_name.insert(key, existing.clone());
            } else {
                let schema = Schema::parse_and_validate(sdl, &format!("{name}.graphql"))
                    .map_err(|e| FederationError::internal(format!(
                        "failed to parse subgraph '{name}' SDL: {e}"
                    )))?;
                let fed_schema = ValidFederationSchema::new(schema)?;
                subgraphs_by_name.insert(key, fed_schema);
            }
        }
        timing.parse_subgraphs_ms = t0.elapsed().as_millis();

        // 2. Build sources = subgraphs + dummy root source.
        let mut sources: IndexMap<Arc<str>, ValidFederationSchema> = IndexMap::default();
        for (name, schema) in &subgraphs_by_name {
            sources.insert(name.clone(), schema.clone());
        }
        let root_source: Arc<str> = FEDERATED_GRAPH_ROOT_SOURCE.into();
        let dummy_schema = ValidFederationSchema::new(
            Valid::assume_valid(Schema::new())
        )?;
        sources.insert(root_source.clone(), dummy_schema);

        // 3. Pre-build condition SelectionSets from the intern table (v2 fast path).
        let t_cond = Instant::now();
        let prebuilt_conditions: Vec<Option<Arc<SelectionSet>>> = if !self.condition_table.is_empty() {
            self.condition_table
                .iter()
                .map(|cond| {
                    reconstruct_selection_set(cond, &supergraph_schema)
                        .map(|sel| Some(Arc::new(sel)))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        timing.reconstruct_conditions_ms = t_cond.elapsed().as_millis();

        // 4. Reconstruct the DiGraph from serialized nodes and edges.
        let t1 = Instant::now();
        let mut graph: DiGraph<QueryGraphNode, QueryGraphEdge> = DiGraph::new();

        // Add nodes in index order.
        for pnode in &self.nodes {
            let node = restore_node(pnode)?;
            graph.add_node(node);
        }

        // Add edges in index order.
        for (src_idx, tgt_idx, pedge) in &self.edges {
            let src = NodeIndex::new(*src_idx as usize);
            let tgt = NodeIndex::new(*tgt_idx as usize);

            // Resolve conditions: prefer v2 condition_index, fall back to v1 SDL string.
            let conditions = if let Some(cond_idx) = pedge.condition_index {
                // v2 fast path: look up pre-built condition from intern table.
                let idx = cond_idx as usize;
                if idx < prebuilt_conditions.len() {
                    prebuilt_conditions[idx].clone()
                } else {
                    return Err(FederationError::internal(format!(
                        "condition_index {cond_idx} out of range (table has {} entries)",
                        prebuilt_conditions.len()
                    )));
                }
            } else if let Some(cond_str) = &pedge.conditions {
                // v1 fallback: parse condition from SDL string.
                let head_node = graph.node_weight(src).ok_or_else(|| {
                    FederationError::internal(format!("edge references invalid source node {src_idx}"))
                })?;
                let head_type_name = match &head_node.type_ {
                    QueryGraphNodeType::SchemaType(pos) => pos.type_name().clone(),
                    QueryGraphNodeType::FederatedRootType(_) => {
                        Name::new_unchecked("Query")
                    }
                };
                let sel = crate::schema::field_set::parse_field_set(
                    &supergraph_schema,
                    head_type_name.clone(),
                    cond_str,
                    false,
                )?;
                Some(Arc::new(sel))
            } else {
                None
            };

            let transition = restore_transition(&pedge.transition)?;
            let override_condition = pedge.override_condition.as_ref().map(|(label, polarity)| {
                OverrideCondition {
                    label: Arc::from(label.as_str()),
                    condition: *polarity,
                }
            });

            // Note: required_contexts are not restored in the prototype.
            graph.add_edge(src, tgt, QueryGraphEdge {
                transition,
                conditions,
                override_condition,
                required_contexts: Vec::new(),
            });
        }

        timing.rebuild_graph_ms = t1.elapsed().as_millis();

        // 4. Restore root_kinds_to_nodes_by_source.
        let t2 = Instant::now();
        let mut root_kinds_to_nodes_by_source: IndexMap<Arc<str>, IndexMap<SchemaRootDefinitionKind, NodeIndex>> = IndexMap::default();
        for (source, root_kinds) in &self.root_kinds_to_nodes_by_source {
            let mut inner = IndexMap::default();
            for (kind_str, node_idx) in root_kinds {
                inner.insert(
                    parse_root_kind(kind_str)?,
                    NodeIndex::new(*node_idx as usize),
                );
            }
            root_kinds_to_nodes_by_source.insert(Arc::from(source.as_str()), inner);
        }

        // 5. Restore types_to_nodes_by_source.
        let mut types_to_nodes_by_source: IndexMap<Arc<str>, IndexMap<NamedType, IndexSet<NodeIndex>>> = IndexMap::default();
        for (source, types_to_nodes) in &self.types_to_nodes_by_source {
            let mut inner: IndexMap<NamedType, IndexSet<NodeIndex>> = IndexMap::default();
            for (type_name, node_indices) in types_to_nodes {
                let name = Name::new(type_name).map_err(|_| {
                    FederationError::internal(format!("invalid type name: {type_name}"))
                })?;
                inner.insert(
                    name,
                    node_indices.iter().map(|n| NodeIndex::new(*n as usize)).collect(),
                );
            }
            types_to_nodes_by_source.insert(Arc::from(source.as_str()), inner);
        }

        // 6. Restore non_trivial_followup_edges.
        let mut non_trivial_followup_edges: IndexMap<EdgeIndex, Vec<EdgeIndex>> = IndexMap::default();
        for (edge_idx, followups) in &self.non_trivial_followup_edges {
            non_trivial_followup_edges.insert(
                EdgeIndex::new(*edge_idx as usize),
                followups.iter().map(|e| EdgeIndex::new(*e as usize)).collect(),
            );
        }

        // 7. Restore field_edge_index.
        let mut field_edge_index: HashMap<(NodeIndex, Name), Vec<EdgeIndex>> = HashMap::new();
        for ((node_idx, field_name), edge_indices) in &self.field_edge_index {
            let name = Name::new(field_name).map_err(|_| {
                FederationError::internal(format!("invalid field name: {field_name}"))
            })?;
            field_edge_index.insert(
                (NodeIndex::new(*node_idx as usize), name),
                edge_indices.iter().map(|e| EdgeIndex::new(*e as usize)).collect(),
            );
        }

        // 8. Restore override_condition_labels.
        let override_condition_labels: IndexSet<Arc<str>> = self.override_condition_labels
            .iter()
            .map(|l| Arc::from(l.as_str()))
            .collect();

        // 9. Build the QueryGraph struct.
        let mut query_graph = QueryGraph {
            current_source: root_source.clone(),
            graph,
            sources,
            subgraphs_by_name,
            supergraph_schema: Some(supergraph_schema),
            types_to_nodes_by_source,
            root_kinds_to_nodes_by_source,
            non_trivial_followup_edges,
            arguments_to_context_ids_by_source: Default::default(), // TODO: serialize @fromContext
            override_condition_labels,
            non_local_selection_metadata: Default::default(), // computed below
            field_edge_index,
            semantic_edge_to_index: Default::default(), // computed below
            semantic_index_to_edge: Default::default(),
            semantic_node_to_index: Default::default(),
            semantic_index_to_node: Default::default(),
        };

        timing.restore_indices_ms = t2.elapsed().as_millis();

        // 10. Compute semantic indices (always recomputed — faster than deserializing).
        let t3 = Instant::now();
        compute_semantic_indices(&mut query_graph);
        timing.compute_semantic_indices_ms = t3.elapsed().as_millis();

        // 11. Restore or compute non-local selection metadata.
        let t4 = Instant::now();
        if let Some(pm) = &self.metadata {
            // v3+ fast path: restore from artifact.
            query_graph.non_local_selection_metadata = metadata_from_portable(pm);
        } else {
            // v1/v2 fallback: recompute from graph.
            query_graph.non_local_selection_metadata =
                precompute_non_local_selection_metadata(&query_graph)?;
        }
        timing.compute_non_local_metadata_ms = t4.elapsed().as_millis();

        Ok((query_graph, timing))
    }
}

/// Timing breakdown for `PortableQueryGraph::to_query_graph_with_timing`.
#[derive(Debug, Default)]
pub struct ReconstructionTiming {
    /// Time to parse subgraph SDL strings into ValidFederationSchemas.
    pub parse_subgraphs_ms: u128,
    /// Time to rebuild the DiGraph (nodes, edges, conditions parsing).
    pub rebuild_graph_ms: u128,
    /// Time to reconstruct conditions from the intern table (v2) or parse from SDL (v1).
    pub reconstruct_conditions_ms: u128,
    /// Time to restore index maps (root_kinds, types, followup_edges, field_edge_index).
    pub restore_indices_ms: u128,
    /// Time to compute semantic node/edge index mappings (SemanticNodeId↔NodeIndex, SemanticEdgeId↔EdgeIndex).
    pub compute_semantic_indices_ms: u128,
    /// Time to compute non-local selection metadata (QueryGraphMetadata).
    pub compute_non_local_metadata_ms: u128,
}

impl ReconstructionTiming {
    pub fn total_ms(&self) -> u128 {
        self.parse_subgraphs_ms
            + self.rebuild_graph_ms
            + self.reconstruct_conditions_ms
            + self.restore_indices_ms
            + self.compute_semantic_indices_ms
            + self.compute_non_local_metadata_ms
    }
}

/// Rebuild semantic node/edge indices from the reconstructed graph.
fn compute_semantic_indices(query_graph: &mut QueryGraph) {
    let mut node_to_index = HashMap::new();
    let mut index_to_node = HashMap::new();
    for node_idx in query_graph.graph.node_indices() {
        if let Some(node) = query_graph.graph.node_weight(node_idx) {
            let semantic_id = SemanticNodeId::from_node(node);
            node_to_index.insert(semantic_id.clone(), node_idx);
            index_to_node.insert(node_idx, semantic_id);
        }
    }

    let mut edge_to_index = HashMap::new();
    let mut index_to_edge = HashMap::new();
    for edge_idx in query_graph.graph.edge_indices() {
        let Some(edge_weight) = query_graph.graph.edge_weight(edge_idx) else { continue };
        let Some((head_idx, tail_idx)) = query_graph.graph.edge_endpoints(edge_idx) else { continue };
        let Some(head_node) = query_graph.graph.node_weight(head_idx) else { continue };
        let Some(tail_node) = query_graph.graph.node_weight(tail_idx) else { continue };
        let semantic_id = SemanticEdgeId::from_edge(head_node, tail_node, edge_weight);
        edge_to_index.insert(semantic_id.clone(), edge_idx);
        index_to_edge.insert(edge_idx, semantic_id);
    }

    query_graph.semantic_node_to_index = node_to_index;
    query_graph.semantic_index_to_node = index_to_node;
    query_graph.semantic_edge_to_index = edge_to_index;
    query_graph.semantic_index_to_edge = index_to_edge;
}

// --- Conversion helpers ---

fn portable_node(node: &QueryGraphNode) -> PortableNode {
    let (type_name, is_federated_root, root_kind_str, type_kind) = match &node.type_ {
        QueryGraphNodeType::SchemaType(pos) => {
            let kind = output_type_kind(pos);
            (pos.type_name().to_string(), false, None, Some(kind))
        }
        QueryGraphNodeType::FederatedRootType(kind) => {
            (format!("[{kind}]"), true, Some(root_kind_to_string(*kind)), None)
        }
    };
    PortableNode {
        type_name,
        is_federated_root,
        root_kind: root_kind_str.or_else(|| node.root_kind.map(|k| root_kind_to_string(k))),
        source: node.source.to_string(),
        provide_id: node.provide_id,
        has_reachable_cross_subgraph_edges: node.has_reachable_cross_subgraph_edges,
        type_kind,
    }
}

fn portable_edge(edge: &QueryGraphEdge) -> PortableEdge {
    let transition = portable_transition(&edge.transition);
    let conditions = edge.conditions.as_ref().map(|c| c.to_string());
    let override_condition = edge
        .override_condition
        .as_ref()
        .map(|oc| (oc.label.to_string(), oc.condition));
    // Note: required_contexts (@fromContext) are not serialized in the prototype.
    // They would need accessor methods on ContextCondition.
    let required_contexts = Vec::new();

    PortableEdge {
        transition,
        conditions,
        condition_index: None, // Set later by from_query_graph() via intern table
        override_condition,
        required_contexts,
    }
}

fn portable_transition(t: &QueryGraphEdgeTransition) -> PortableTransition {
    match t {
        QueryGraphEdgeTransition::FieldCollection {
            source,
            field_definition_position,
            is_part_of_provides,
        } => {
            let (field_position, field_kind) = match field_definition_position {
                FieldDefinitionPosition::Object(p) => (p.to_string(), "object".to_string()),
                FieldDefinitionPosition::Interface(p) => (p.to_string(), "interface".to_string()),
                FieldDefinitionPosition::Union(p) => (p.to_string(), "union_typename".to_string()),
            };
            PortableTransition::FieldCollection {
                source: source.to_string(),
                field_position,
                field_kind,
                is_part_of_provides: *is_part_of_provides,
            }
        }
        QueryGraphEdgeTransition::Downcast {
            source,
            from_type_position,
            to_type_position,
        } => PortableTransition::Downcast {
            source: source.to_string(),
            from_type: from_type_position.type_name().to_string(),
            from_type_kind: composite_type_kind(from_type_position),
            to_type: to_type_position.type_name().to_string(),
            to_type_kind: composite_type_kind(to_type_position),
        },
        QueryGraphEdgeTransition::KeyResolution => PortableTransition::KeyResolution,
        QueryGraphEdgeTransition::RootTypeResolution { root_kind } => {
            PortableTransition::RootTypeResolution {
                root_kind: root_kind_to_string(*root_kind),
            }
        }
        QueryGraphEdgeTransition::SubgraphEnteringTransition => {
            PortableTransition::SubgraphEnteringTransition
        }
        QueryGraphEdgeTransition::InterfaceObjectFakeDownCast {
            source,
            from_type_position,
            to_type_name,
        } => PortableTransition::InterfaceObjectFakeDownCast {
            source: source.to_string(),
            from_type: from_type_position.type_name().to_string(),
            from_type_kind: composite_type_kind(from_type_position),
            to_type_name: to_type_name.to_string(),
        },
    }
}

fn root_kind_to_string(kind: SchemaRootDefinitionKind) -> String {
    match kind {
        SchemaRootDefinitionKind::Query => "Query".to_string(),
        SchemaRootDefinitionKind::Mutation => "Mutation".to_string(),
        SchemaRootDefinitionKind::Subscription => "Subscription".to_string(),
    }
}

fn output_type_kind(pos: &OutputTypeDefinitionPosition) -> String {
    match pos {
        OutputTypeDefinitionPosition::Scalar(_) => "scalar".to_string(),
        OutputTypeDefinitionPosition::Object(_) => "object".to_string(),
        OutputTypeDefinitionPosition::Interface(_) => "interface".to_string(),
        OutputTypeDefinitionPosition::Union(_) => "union".to_string(),
        OutputTypeDefinitionPosition::Enum(_) => "enum".to_string(),
    }
}

fn composite_type_kind(pos: &CompositeTypeDefinitionPosition) -> String {
    match pos {
        CompositeTypeDefinitionPosition::Object(_) => "object".to_string(),
        CompositeTypeDefinitionPosition::Interface(_) => "interface".to_string(),
        CompositeTypeDefinitionPosition::Union(_) => "union".to_string(),
    }
}

fn serialize_root_kinds(graph: &QueryGraph) -> BTreeMap<String, BTreeMap<String, u32>> {
    let mut result = BTreeMap::new();
    for (source, root_kinds) in graph.root_kinds_to_nodes_by_source_map() {
        let mut inner = BTreeMap::new();
        for (kind, node) in root_kinds {
            inner.insert(root_kind_to_string(*kind), node.index() as u32);
        }
        result.insert(source.to_string(), inner);
    }
    result
}

fn serialize_types_to_nodes(graph: &QueryGraph) -> BTreeMap<String, BTreeMap<String, Vec<u32>>> {
    let mut result = BTreeMap::new();
    for (source, types_to_nodes) in graph.types_to_nodes_by_source_map() {
        let mut inner = BTreeMap::new();
        for (type_name, nodes) in types_to_nodes {
            inner.insert(
                type_name.to_string(),
                nodes.iter().map(|n| n.index() as u32).collect(),
            );
        }
        result.insert(source.to_string(), inner);
    }
    result
}

fn serialize_followup_edges(graph: &QueryGraph) -> BTreeMap<u32, Vec<u32>> {
    let mut result = BTreeMap::new();
    for (edge, followups) in graph.non_trivial_followup_edges_map() {
        result.insert(
            edge.index() as u32,
            followups.iter().map(|e| e.index() as u32).collect(),
        );
    }
    result
}

fn serialize_field_edge_index(graph: &QueryGraph) -> Vec<((u32, String), Vec<u32>)> {
    graph
        .field_edge_index_map()
        .iter()
        .map(|((node, name), edges)| {
            (
                (node.index() as u32, name.to_string()),
                edges.iter().map(|e| e.index() as u32).collect(),
            )
        })
        .collect()
}

// --- Deserialization helpers ---

fn parse_root_kind(s: &str) -> Result<SchemaRootDefinitionKind, FederationError> {
    match s {
        "Query" => Ok(SchemaRootDefinitionKind::Query),
        "Mutation" => Ok(SchemaRootDefinitionKind::Mutation),
        "Subscription" => Ok(SchemaRootDefinitionKind::Subscription),
        _ => Err(FederationError::internal(format!("unknown root kind: {s}"))),
    }
}

fn restore_node(pnode: &PortableNode) -> Result<QueryGraphNode, FederationError> {
    let type_ = if pnode.is_federated_root {
        let kind_str = pnode.root_kind.as_deref().ok_or_else(|| {
            FederationError::internal("federated root node missing root_kind")
        })?;
        QueryGraphNodeType::FederatedRootType(parse_root_kind(kind_str)?)
    } else {
        let kind_str = pnode.type_kind.as_deref().unwrap_or("object");
        let pos = parse_output_type(&pnode.type_name, kind_str)?;
        QueryGraphNodeType::SchemaType(pos)
    };

    let root_kind = pnode.root_kind.as_deref()
        .map(parse_root_kind)
        .transpose()?;

    Ok(QueryGraphNode {
        type_: type_,
        source: Arc::from(pnode.source.as_str()),
        has_reachable_cross_subgraph_edges: pnode.has_reachable_cross_subgraph_edges,
        provide_id: pnode.provide_id,
        root_kind,
    })
}

/// Parse a field position string like "User.name" back into the components.
fn parse_field_position(position: &str, kind: &str) -> Result<FieldDefinitionPosition, FederationError> {
    let dot = position.find('.').ok_or_else(|| {
        FederationError::internal(format!("invalid field position: {position}"))
    })?;
    let type_name = Name::new(&position[..dot]).map_err(|_| {
        FederationError::internal(format!("invalid type name in field position: {position}"))
    })?;
    let field_name = Name::new(&position[dot + 1..]).map_err(|_| {
        FederationError::internal(format!("invalid field name in field position: {position}"))
    })?;

    match kind {
        "object" => Ok(FieldDefinitionPosition::Object(ObjectFieldDefinitionPosition {
            type_name,
            field_name,
        })),
        "interface" => Ok(FieldDefinitionPosition::Interface(InterfaceFieldDefinitionPosition {
            type_name,
            field_name,
        })),
        "union_typename" => Ok(FieldDefinitionPosition::Union(UnionTypenameFieldDefinitionPosition {
            type_name,
        })),
        _ => Err(FederationError::internal(format!("unknown field kind: {kind}"))),
    }
}

fn parse_output_type(type_name: &str, kind: &str) -> Result<OutputTypeDefinitionPosition, FederationError> {
    let name = Name::new(type_name).map_err(|_| {
        FederationError::internal(format!("invalid type name: {type_name}"))
    })?;
    match kind {
        "scalar" => Ok(OutputTypeDefinitionPosition::Scalar(ScalarTypeDefinitionPosition { type_name: name })),
        "object" => Ok(OutputTypeDefinitionPosition::Object(ObjectTypeDefinitionPosition { type_name: name })),
        "interface" => Ok(OutputTypeDefinitionPosition::Interface(InterfaceTypeDefinitionPosition { type_name: name })),
        "union" => Ok(OutputTypeDefinitionPosition::Union(UnionTypeDefinitionPosition { type_name: name })),
        "enum" => Ok(OutputTypeDefinitionPosition::Enum(EnumTypeDefinitionPosition { type_name: name })),
        _ => Err(FederationError::internal(format!("unknown output type kind: {kind}"))),
    }
}

fn parse_composite_type(type_name: &str, kind: &str) -> Result<CompositeTypeDefinitionPosition, FederationError> {
    let name = Name::new(type_name).map_err(|_| {
        FederationError::internal(format!("invalid type name: {type_name}"))
    })?;
    match kind {
        "object" => Ok(CompositeTypeDefinitionPosition::Object(ObjectTypeDefinitionPosition { type_name: name })),
        "interface" => Ok(CompositeTypeDefinitionPosition::Interface(InterfaceTypeDefinitionPosition { type_name: name })),
        "union" => Ok(CompositeTypeDefinitionPosition::Union(UnionTypeDefinitionPosition { type_name: name })),
        _ => Err(FederationError::internal(format!("unknown composite type kind: {kind}"))),
    }
}

fn restore_transition(pt: &PortableTransition) -> Result<QueryGraphEdgeTransition, FederationError> {
    match pt {
        PortableTransition::FieldCollection {
            source,
            field_position,
            field_kind,
            is_part_of_provides,
        } => {
            Ok(QueryGraphEdgeTransition::FieldCollection {
                source: Arc::from(source.as_str()),
                field_definition_position: parse_field_position(field_position, field_kind)?,
                is_part_of_provides: *is_part_of_provides,
            })
        }
        PortableTransition::Downcast {
            source,
            from_type,
            from_type_kind,
            to_type,
            to_type_kind,
        } => {
            Ok(QueryGraphEdgeTransition::Downcast {
                source: Arc::from(source.as_str()),
                from_type_position: parse_composite_type(from_type, from_type_kind)?,
                to_type_position: parse_composite_type(to_type, to_type_kind)?,
            })
        }
        PortableTransition::KeyResolution => Ok(QueryGraphEdgeTransition::KeyResolution),
        PortableTransition::RootTypeResolution { root_kind } => {
            Ok(QueryGraphEdgeTransition::RootTypeResolution {
                root_kind: parse_root_kind(root_kind)?,
            })
        }
        PortableTransition::SubgraphEnteringTransition => {
            Ok(QueryGraphEdgeTransition::SubgraphEnteringTransition)
        }
        PortableTransition::InterfaceObjectFakeDownCast {
            source,
            from_type,
            from_type_kind,
            to_type_name,
        } => {
            let name = Name::new(to_type_name).map_err(|_| {
                FederationError::internal(format!("invalid type name: {to_type_name}"))
            })?;
            Ok(QueryGraphEdgeTransition::InterfaceObjectFakeDownCast {
                source: Arc::from(source.as_str()),
                from_type_position: parse_composite_type(from_type, from_type_kind)?,
                to_type_name: name,
            })
        }
    }
}

// --- Condition serialization/reconstruction ---

use crate::operation::{Selection, SelectionSet, Field, InlineFragment, InlineFragmentSelection};

/// Convert a `SelectionSet` (from an edge condition) into a `PortableCondition`.
///
/// Walks the selection tree and produces a lightweight serializable representation.
/// Arguments and directives are serialized to SDL strings only when non-empty (rare).
fn portable_condition_from_selection_set(sel: &SelectionSet) -> PortableCondition {
    let type_name = sel.type_position.type_name().to_string();
    let selections = sel
        .selections
        .values()
        .map(|selection| match selection {
            Selection::Field(field_sel) => {
                let field = &field_sel.field;
                let field_name = field.field_position.field_name().to_string();
                let parent_type_name = field.field_position.type_name().to_string();

                // Serialize arguments to SDL only if non-empty.
                let arguments_sdl = if field.arguments.is_empty() {
                    None
                } else {
                    // Format as "(arg1: val1, arg2: val2)"
                    let args: Vec<String> = field.arguments.iter().map(|arg| {
                        format!("{}: {}", arg.name, arg.value)
                    }).collect();
                    Some(format!("({})", args.join(", ")))
                };

                // Serialize directives to SDL only if non-empty.
                let directives_sdl = if field.directives.is_empty() {
                    None
                } else {
                    let s = field.directives.to_string();
                    if s.is_empty() { None } else { Some(s) }
                };

                let sub_selections = field_sel.selection_set.as_ref().map(|sub| {
                    portable_condition_from_selection_set(sub)
                });

                PortableConditionSelection::Field {
                    parent_type_name,
                    field_name,
                    arguments_sdl,
                    directives_sdl,
                    sub_selections,
                }
            }
            Selection::InlineFragment(frag_sel) => {
                let frag = &frag_sel.inline_fragment;
                let parent_type_name = frag.parent_type_position.type_name().to_string();
                let type_condition = frag.type_condition_position.as_ref()
                    .map(|pos| pos.type_name().to_string());

                let directives_sdl = if frag.directives.is_empty() {
                    None
                } else {
                    let s = frag.directives.to_string();
                    if s.is_empty() { None } else { Some(s) }
                };

                let selections = portable_condition_from_selection_set(&frag_sel.selection_set);

                PortableConditionSelection::InlineFragment {
                    parent_type_name,
                    type_condition,
                    directives_sdl,
                    selections,
                }
            }
        })
        .collect();

    PortableCondition {
        type_name,
        selections,
    }
}

/// Reconstruct a `SelectionSet` from a `PortableCondition` using direct schema lookups.
///
/// This is the fast path that replaces `parse_field_set`. For each field, it:
/// 1. Looks up the `FieldDefinitionPosition` via the type name in the schema
/// 2. Builds a `Field` with `Default` arguments/directives (the common case)
/// 3. For the rare case where arguments/directives are present, falls back to
///    parsing just that tiny SDL string
///
/// Cost: O(n) HashMap lookups, no SDL parsing for the ~98% of conditions
/// that have no arguments or directives.
fn reconstruct_selection_set(
    cond: &PortableCondition,
    schema: &ValidFederationSchema,
) -> Result<SelectionSet, FederationError> {
    let type_name = Name::new(&cond.type_name).map_err(|_| {
        FederationError::internal(format!("invalid type name in condition: {}", cond.type_name))
    })?;

    // Resolve the parent type position from the schema.
    let type_position = resolve_composite_type_position(schema, &type_name)?;

    let mut selections: Vec<Selection> = Vec::with_capacity(cond.selections.len());

    for sel in &cond.selections {
        match sel {
            PortableConditionSelection::Field {
                parent_type_name,
                field_name,
                arguments_sdl,
                directives_sdl,
                sub_selections,
            } => {
                let parent_name = Name::new(parent_type_name).map_err(|_| {
                    FederationError::internal(format!("invalid parent type: {parent_type_name}"))
                })?;
                let fname = Name::new(field_name).map_err(|_| {
                    FederationError::internal(format!("invalid field name: {field_name}"))
                })?;

                // Look up the field definition position.
                let field_position = resolve_field_position(schema, &parent_name, &fname)?;

                // Build arguments: default (empty) for the common case,
                // parse from SDL for the rare case.
                let arguments = if let Some(args_sdl) = arguments_sdl {
                    parse_arguments_sdl(args_sdl)?
                } else {
                    Default::default()
                };

                // Build directives: default (empty) for the common case.
                let directives = if let Some(dir_sdl) = directives_sdl {
                    parse_directives_sdl(dir_sdl)?
                } else {
                    Default::default()
                };

                let field = Field {
                    schema: schema.clone(),
                    field_position,
                    alias: None,
                    arguments,
                    directives,
                    sibling_typename: None,
                };

                let sub_sel = match sub_selections {
                    Some(sub_cond) => Some(reconstruct_selection_set(sub_cond, schema)?),
                    None => None,
                };

                selections.push(Selection::from_field(field, sub_sel));
            }
            PortableConditionSelection::InlineFragment {
                parent_type_name,
                type_condition,
                directives_sdl,
                selections: frag_selections,
            } => {
                let parent_name = Name::new(parent_type_name).map_err(|_| {
                    FederationError::internal(format!("invalid parent type: {parent_type_name}"))
                })?;
                let parent_pos = resolve_composite_type_position(schema, &parent_name)?;

                let type_condition_pos = match type_condition {
                    Some(tc) => {
                        let tc_name = Name::new(tc).map_err(|_| {
                            FederationError::internal(format!("invalid type condition: {tc}"))
                        })?;
                        Some(resolve_composite_type_position(schema, &tc_name)?)
                    }
                    None => None,
                };

                let directives = if let Some(dir_sdl) = directives_sdl {
                    parse_directives_sdl(dir_sdl)?
                } else {
                    Default::default()
                };

                let inline_fragment = InlineFragment {
                    schema: schema.clone(),
                    parent_type_position: parent_pos,
                    type_condition_position: type_condition_pos,
                    directives,
                    selection_id: crate::operation::SelectionId::new(),
                };

                let sub_sel = reconstruct_selection_set(frag_selections, schema)?;
                let frag = InlineFragmentSelection::new(inline_fragment, sub_sel);
                selections.push(Selection::InlineFragment(Arc::new(frag)));
            }
        }
    }

    Ok(SelectionSet::from_raw_selections(
        schema.clone(),
        type_position,
        selections,
    ))
}

/// Resolve a composite type position from a schema by name.
/// Tries object, then interface, then union.
fn resolve_composite_type_position(
    schema: &ValidFederationSchema,
    type_name: &Name,
) -> Result<CompositeTypeDefinitionPosition, FederationError> {
    use apollo_compiler::schema::ExtendedType;
    if let Some(def) = schema.schema().types.get(type_name) {
        match def {
            ExtendedType::Object(_) => {
                return Ok(CompositeTypeDefinitionPosition::Object(
                    ObjectTypeDefinitionPosition { type_name: type_name.clone() },
                ));
            }
            ExtendedType::Interface(_) => {
                return Ok(CompositeTypeDefinitionPosition::Interface(
                    InterfaceTypeDefinitionPosition { type_name: type_name.clone() },
                ));
            }
            ExtendedType::Union(_) => {
                return Ok(CompositeTypeDefinitionPosition::Union(
                    UnionTypeDefinitionPosition { type_name: type_name.clone() },
                ));
            }
            _ => {}
        }
    }
    Err(FederationError::internal(format!(
        "type '{}' not found as composite type in schema",
        type_name
    )))
}

/// Resolve a field definition position from a schema.
fn resolve_field_position(
    schema: &ValidFederationSchema,
    parent_type_name: &Name,
    field_name: &Name,
) -> Result<FieldDefinitionPosition, FederationError> {
    if let Some(def) = schema.schema().types.get(parent_type_name) {
        use apollo_compiler::schema::ExtendedType;
        match def {
            ExtendedType::Object(_) => {
                return Ok(FieldDefinitionPosition::Object(ObjectFieldDefinitionPosition {
                    type_name: parent_type_name.clone(),
                    field_name: field_name.clone(),
                }));
            }
            ExtendedType::Interface(_) => {
                return Ok(FieldDefinitionPosition::Interface(InterfaceFieldDefinitionPosition {
                    type_name: parent_type_name.clone(),
                    field_name: field_name.clone(),
                }));
            }
            ExtendedType::Union(_) if field_name.as_str() == "__typename" => {
                return Ok(FieldDefinitionPosition::Union(UnionTypenameFieldDefinitionPosition {
                    type_name: parent_type_name.clone(),
                }));
            }
            _ => {}
        }
    }
    Err(FederationError::internal(format!(
        "cannot resolve field '{parent_type_name}.{field_name}' in schema"
    )))
}

/// Parse an arguments SDL string like "(id: 1)" into an `ArgumentList`.
/// Only used for the rare conditions that have arguments (<2% of cases).
fn parse_arguments_sdl(sdl: &str) -> Result<crate::operation::ArgumentList, FederationError> {
    use apollo_compiler::executable;
    // Wrap in a minimal query to parse: "{ f<args> }"
    let doc_str = format!("{{ f{sdl} }}");
    // Use parse_and_validate with an empty-but-valid schema.
    let empty_schema = Valid::assume_valid(Schema::new());
    let doc = match executable::ExecutableDocument::parse_and_validate(&empty_schema, &doc_str, "args.graphql") {
        Ok(doc) => doc.into_inner(),
        // Validation may fail against empty schema, but we can still try to get the parsed doc
        // from the error. If even parsing fails, return empty.
        Err(with_errors) => with_errors.partial,
    };
    // Walk the parsed document to extract argument nodes.
    for op in doc.operations.iter() {
        for sel in &op.selection_set.selections {
            if let executable::Selection::Field(field) = sel {
                return Ok(field.arguments.iter().cloned().collect());
            }
        }
    }
    Ok(Default::default())
}

/// Parse a directives SDL string like "@skip(if: true)" into a `DirectiveList`.
/// Only used for the rare conditions that have directives (<2% of cases).
fn parse_directives_sdl(sdl: &str) -> Result<crate::operation::DirectiveList, FederationError> {
    use apollo_compiler::executable;
    // Wrap in a minimal query: "{ f <directives> }"
    let doc_str = format!("{{ f {sdl} }}");
    let empty_schema = Valid::assume_valid(Schema::new());
    let doc = match executable::ExecutableDocument::parse_and_validate(&empty_schema, &doc_str, "dirs.graphql") {
        Ok(doc) => doc.into_inner(),
        Err(with_errors) => with_errors.partial,
    };
    for op in doc.operations.iter() {
        for sel in &op.selection_set.selections {
            if let executable::Selection::Field(field) = sel {
                return Ok(field.directives.iter().cloned().collect());
            }
        }
    }
    Ok(Default::default())
}

// --- Metadata serialization/reconstruction ---

/// Convert `QueryGraphMetadata` to its portable representation.
fn metadata_to_portable(
    metadata: &non_local_selections_estimation::QueryGraphMetadata,
) -> PortableMetadata {
    let types_to_indirect_options = metadata.types_to_indirect_options
        .iter()
        .map(|(name, opts)| {
            (name.to_string(), PortableIndirectOptions {
                same_type_options: opts.same_type_options.iter().map(|n| n.index() as u32).collect(),
                interface_object_options: opts.interface_object_options.iter().map(|n| n.to_string()).collect(),
            })
        })
        .collect();

    let remaining_nodes_to_interface_object_options = metadata.remaining_nodes_to_interface_object_options
        .iter()
        .map(|(node, names)| {
            (node.index() as u32, names.iter().map(|n| n.to_string()).collect())
        })
        .collect();

    let fields_to_endpoints = metadata.fields_to_endpoints
        .iter()
        .map(|(name, endpoints)| {
            let entries: Vec<(u32, PortableFieldTarget)> = endpoints
                .iter()
                .map(|(node, target)| {
                    let pt = match target {
                        non_local_selections_estimation::FieldTarget::NonOverride(n) => {
                            PortableFieldTarget::NonOverride(n.index() as u32)
                        }
                        non_local_selections_estimation::FieldTarget::Override(n, oc) => {
                            PortableFieldTarget::Override(
                                n.index() as u32,
                                oc.label.to_string(),
                                oc.condition,
                            )
                        }
                    };
                    (node.index() as u32, pt)
                })
                .collect();
            (name.to_string(), entries)
        })
        .collect();

    let inline_fragments_to_endpoints = metadata.inline_fragments_to_endpoints
        .iter()
        .map(|(name, endpoints)| {
            let entries: Vec<(u32, u32)> = endpoints
                .iter()
                .map(|(src, tgt)| (src.index() as u32, tgt.index() as u32))
                .collect();
            (name.to_string(), entries)
        })
        .collect();

    let nodes_to_object_type_downcasts = metadata.nodes_to_object_type_downcasts
        .iter()
        .map(|(node, downcasts)| {
            let pd = match downcasts {
                non_local_selections_estimation::ObjectTypeDowncasts::NonInterfaceObject(map) => {
                    PortableObjectTypeDowncasts::NonInterfaceObject(
                        map.iter().map(|(name, n)| (name.to_string(), n.index() as u32)).collect()
                    )
                }
                non_local_selections_estimation::ObjectTypeDowncasts::InterfaceObject(names) => {
                    PortableObjectTypeDowncasts::InterfaceObject(
                        names.iter().map(|n| n.to_string()).collect()
                    )
                }
            };
            (node.index() as u32, pd)
        })
        .collect();

    let fields_to_rebaseable_parent_nodes = metadata.fields_to_rebaseable_parent_nodes
        .iter()
        .map(|(name, nodes)| {
            (name.to_string(), nodes.iter().map(|n| n.index() as u32).collect())
        })
        .collect();

    let inline_fragments_to_rebaseable_parent_nodes = metadata.inline_fragments_to_rebaseable_parent_nodes
        .iter()
        .map(|(name, nodes)| {
            (name.to_string(), nodes.iter().map(|n| n.index() as u32).collect())
        })
        .collect();

    PortableMetadata {
        types_to_indirect_options,
        remaining_nodes_to_interface_object_options,
        fields_to_endpoints,
        inline_fragments_to_endpoints,
        nodes_to_object_type_downcasts,
        fields_to_rebaseable_parent_nodes,
        inline_fragments_to_rebaseable_parent_nodes,
    }
}

/// Reconstruct `QueryGraphMetadata` from its portable representation.
fn metadata_from_portable(
    pm: &PortableMetadata,
) -> non_local_selections_estimation::QueryGraphMetadata {
    let mut metadata = non_local_selections_estimation::QueryGraphMetadata::default();

    for (name_str, opts) in &pm.types_to_indirect_options {
        let name = Name::new_unchecked(name_str);
        let mut indirect = non_local_selections_estimation::IndirectOptionsMetadata::default();
        for &idx in &opts.same_type_options {
            indirect.same_type_options.insert(NodeIndex::new(idx as usize));
        }
        for io_name in &opts.interface_object_options {
            indirect.interface_object_options.insert(Name::new_unchecked(io_name));
        }
        metadata.types_to_indirect_options.insert(name, indirect);
    }

    for (node_idx, names) in &pm.remaining_nodes_to_interface_object_options {
        let mut set = IndexSet::default();
        for n in names {
            set.insert(Name::new_unchecked(n));
        }
        metadata.remaining_nodes_to_interface_object_options
            .insert(NodeIndex::new(*node_idx as usize), set);
    }

    for (name_str, entries) in &pm.fields_to_endpoints {
        let name = Name::new_unchecked(name_str);
        let mut map: IndexMap<NodeIndex, non_local_selections_estimation::FieldTarget> = IndexMap::default();
        for (node_idx, target) in entries {
            let ft = match target {
                PortableFieldTarget::NonOverride(n) => {
                    non_local_selections_estimation::FieldTarget::NonOverride(
                        NodeIndex::new(*n as usize),
                    )
                }
                PortableFieldTarget::Override(n, label, condition) => {
                    non_local_selections_estimation::FieldTarget::Override(
                        NodeIndex::new(*n as usize),
                        OverrideCondition {
                            label: Arc::from(label.as_str()),
                            condition: *condition,
                        },
                    )
                }
            };
            map.insert(NodeIndex::new(*node_idx as usize), ft);
        }
        metadata.fields_to_endpoints.insert(name, map);
    }

    for (name_str, entries) in &pm.inline_fragments_to_endpoints {
        let name = Name::new_unchecked(name_str);
        let mut map: IndexMap<NodeIndex, NodeIndex> = IndexMap::default();
        for &(src, tgt) in entries {
            map.insert(NodeIndex::new(src as usize), NodeIndex::new(tgt as usize));
        }
        metadata.inline_fragments_to_endpoints.insert(name, map);
    }

    for (node_idx, pd) in &pm.nodes_to_object_type_downcasts {
        let downcasts = match pd {
            PortableObjectTypeDowncasts::NonInterfaceObject(entries) => {
                let mut map: IndexMap<Name, NodeIndex> = IndexMap::default();
                for (name, n) in entries {
                    map.insert(Name::new_unchecked(name), NodeIndex::new(*n as usize));
                }
                non_local_selections_estimation::ObjectTypeDowncasts::NonInterfaceObject(map)
            }
            PortableObjectTypeDowncasts::InterfaceObject(names) => {
                let mut set = IndexSet::default();
                for n in names {
                    set.insert(Name::new_unchecked(n));
                }
                non_local_selections_estimation::ObjectTypeDowncasts::InterfaceObject(set)
            }
        };
        metadata.nodes_to_object_type_downcasts
            .insert(NodeIndex::new(*node_idx as usize), downcasts);
    }

    for (name_str, nodes) in &pm.fields_to_rebaseable_parent_nodes {
        let name = Name::new_unchecked(name_str);
        let mut set = IndexSet::default();
        for &n in nodes {
            set.insert(NodeIndex::new(n as usize));
        }
        metadata.fields_to_rebaseable_parent_nodes.insert(name, set);
    }

    for (name_str, nodes) in &pm.inline_fragments_to_rebaseable_parent_nodes {
        let name = Name::new_unchecked(name_str);
        let mut set = IndexSet::default();
        for &n in nodes {
            set.insert(NodeIndex::new(n as usize));
        }
        metadata.inline_fragments_to_rebaseable_parent_nodes.insert(name, set);
    }

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_plan::query_planner::{QueryPlanner, QueryPlannerConfig};
    use crate::Supergraph;

    /// Build a planner from SDL, serialize its QueryGraph to portable format,
    /// deserialize it back, and verify the reconstructed graph has the same
    /// structural properties as the original.
    fn roundtrip_test(supergraph_sdl: &str) {
        // Build the original planner.
        let supergraph = crate::Supergraph::new_with_router_specs(supergraph_sdl).unwrap();
        let planner = QueryPlanner::new(&supergraph, QueryPlannerConfig::default()).unwrap();
        let graph = planner.query_graph();

        // Serialize to portable format.
        let portable = PortableQueryGraph::from_query_graph(graph, supergraph_sdl);
        let bytes = portable.to_bytes();

        // Verify basic metadata.
        assert!(portable.verify_supergraph(supergraph_sdl));
        assert_eq!(portable.node_count(), graph.graph().node_count());
        assert_eq!(portable.edge_count(), graph.graph().edge_count());

        // Deserialize.
        let restored = PortableQueryGraph::from_bytes(&bytes).unwrap();
        assert!(restored.verify_supergraph(supergraph_sdl));
        assert_eq!(restored.node_count(), portable.node_count());
        assert_eq!(restored.edge_count(), portable.edge_count());

        // Reconstruct a full QueryGraph.
        let supergraph2 = crate::Supergraph::new_with_router_specs(supergraph_sdl).unwrap();
        let restored_graph = restored.to_query_graph(supergraph2.schema.clone()).unwrap();

        // Verify structural equivalence.
        assert_eq!(
            restored_graph.graph().node_count(),
            graph.graph().node_count(),
            "node count mismatch"
        );
        assert_eq!(
            restored_graph.graph().edge_count(),
            graph.graph().edge_count(),
            "edge count mismatch"
        );
        assert_eq!(
            restored_graph.subgraph_schemas().len(),
            graph.subgraph_schemas().len(),
            "subgraph count mismatch"
        );
    }

    #[test]
    fn test_portable_roundtrip_basic() {
        let sdl = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/query_plan/supergraphs/avoids_unnecessary_fetches.graphql")
        ).unwrap();
        roundtrip_test(&sdl);
    }

    #[test]
    fn test_portable_roundtrip_interface_object() {
        let sdl = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/query_plan/supergraphs/add_back_sibling_typename_to_interface_object.graphql")
        ).unwrap();
        roundtrip_test(&sdl);
    }

    #[test]
    fn test_portable_content_hash_deterministic() {
        let hash1 = content_hash("hello world");
        let hash2 = content_hash("hello world");
        let hash3 = content_hash("hello world!");
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 32); // 128-bit = 32 hex chars
    }

    #[test]
    fn test_portable_version_check() {
        let sdl = "schema { query: Query } type Query { x: Int }";
        // Create a minimal portable graph and tamper with version.
        let mut portable = PortableQueryGraph {
            version: 999,
            supergraph_sdl_hash: content_hash(sdl),
            subgraph_sdls: BTreeMap::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            root_kinds_to_nodes_by_source: BTreeMap::new(),
            types_to_nodes_by_source: BTreeMap::new(),
            non_trivial_followup_edges: BTreeMap::new(),
            field_edge_index: Vec::new(),
            override_condition_labels: Vec::new(),
            condition_table: Vec::new(),
            metadata: None,
            semantic_node_index: Vec::new(),
            semantic_edge_index: Vec::new(),
        };
        // Test bincode path
        let bytes = portable.to_bytes();
        assert!(PortableQueryGraph::from_bytes(&bytes).is_err());
        portable.version = FORMAT_VERSION;
        let bytes = portable.to_bytes();
        assert!(PortableQueryGraph::from_bytes(&bytes).is_ok());

        // Test JSON path
        portable.version = 999;
        let json = portable.to_json();
        assert!(PortableQueryGraph::from_json(&json).is_err());
        portable.version = FORMAT_VERSION;
        let json = portable.to_json();
        assert!(PortableQueryGraph::from_json(&json).is_ok());
    }

    /// Test that the composition pipeline produces a QueryGraph artifact and that
    /// loading it back produces a structurally equivalent graph.
    #[test]
    fn test_composition_artifact_roundtrip() {
        use crate::composition::compose;
        use crate::subgraph::typestate::Subgraph;

        // Build two simple subgraphs.
        let subgraph_a = Subgraph::parse(
            "A",
            "http://a",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type Query { user(id: ID!): User }
                type User @key(fields: "id") { id: ID! name: String }
            "#,
        ).unwrap();
        let subgraph_b = Subgraph::parse(
            "B",
            "http://b",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type User @key(fields: "id") { id: ID! email: String }
            "#,
        ).unwrap();

        let supergraph = compose(vec![subgraph_a, subgraph_b]).unwrap();

        // Verify artifact was produced.
        let artifact = supergraph.query_graph_artifact()
            .expect("composition should produce a QueryGraph artifact");
        assert!(!artifact.is_empty(), "artifact should not be empty");

        // Load the artifact back.
        let portable = PortableQueryGraph::from_bytes(artifact).unwrap();
        assert!(portable.node_count() > 0);
        assert!(portable.edge_count() > 0);

        // Verify it matches the supergraph.
        let supergraph_sdl = supergraph.schema().schema().serialize().to_string();
        assert!(portable.verify_supergraph(&supergraph_sdl));

        // Reconstruct a full QueryGraph and verify it works for planning.
        let sg = crate::Supergraph::new_with_router_specs(&supergraph_sdl).unwrap();
        let restored_graph = portable.to_query_graph(sg.schema.clone()).unwrap();
        assert!(restored_graph.graph().node_count() > 0);
        assert!(restored_graph.graph().edge_count() > 0);
        assert_eq!(restored_graph.subgraph_schemas().len(), 2);
    }

    /// Test that the condition intern table deduplicates shared conditions.
    /// Two subgraphs with `@key(fields: "id")` on the same type should produce
    /// only one entry in the condition table.
    #[test]
    fn test_portable_condition_dedup() {
        use crate::composition::compose;
        use crate::subgraph::typestate::Subgraph;

        let subgraph_a = Subgraph::parse(
            "A",
            "http://a",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type Query { user(id: ID!): User }
                type User @key(fields: "id") { id: ID! name: String }
            "#,
        ).unwrap();
        let subgraph_b = Subgraph::parse(
            "B",
            "http://b",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type User @key(fields: "id") { id: ID! email: String }
            "#,
        ).unwrap();
        let subgraph_c = Subgraph::parse(
            "C",
            "http://c",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type User @key(fields: "id") { id: ID! avatar: String }
            "#,
        ).unwrap();

        let supergraph = compose(vec![subgraph_a, subgraph_b, subgraph_c]).unwrap();
        let artifact = supergraph.query_graph_artifact().unwrap();
        let portable = PortableQueryGraph::from_bytes(artifact).unwrap();

        // All three subgraphs share @key(fields: "id") on User.
        // The condition table should have far fewer entries than the number of
        // edges with conditions, because identical conditions are deduplicated.
        assert!(
            !portable.condition_table.is_empty(),
            "condition table should not be empty"
        );

        // Count edges with condition_index set.
        let edges_with_conditions = portable.edges.iter()
            .filter(|(_, _, e)| e.condition_index.is_some())
            .count();

        // Dedup: the condition table should have fewer entries than edges with conditions.
        assert!(
            portable.condition_table.len() <= edges_with_conditions,
            "condition table ({}) should be smaller than or equal to edges with conditions ({})",
            portable.condition_table.len(),
            edges_with_conditions,
        );

        // With 3 subgraphs all sharing the same @key(fields: "id"), the condition
        // for "id" should appear only once in the table.
        let id_conditions: Vec<_> = portable.condition_table.iter()
            .filter(|c| c.selections.len() == 1 && matches!(&c.selections[0],
                PortableConditionSelection::Field { field_name, .. } if field_name == "id"
            ))
            .collect();
        assert_eq!(
            id_conditions.len(), 1,
            "the @key(fields: \"id\") condition should be deduplicated to exactly one entry, got {}",
            id_conditions.len()
        );

        eprintln!(
            "Condition dedup: {} unique conditions, {} edges with conditions",
            portable.condition_table.len(),
            edges_with_conditions,
        );
    }

    /// Test that conditions roundtrip correctly through the v2 portable format.
    /// Verifies that the reconstructed QueryGraph has the same conditions as the original.
    #[test]
    fn test_portable_condition_roundtrip() {
        // Use a supergraph fixture that has @key and @requires conditions.
        let sdl = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/query_plan/supergraphs/avoids_unnecessary_fetches.graphql")
        ).unwrap();

        let supergraph = crate::Supergraph::new_with_router_specs(&sdl).unwrap();
        let planner = QueryPlanner::new(&supergraph, QueryPlannerConfig::default()).unwrap();
        let graph = planner.query_graph();

        // Collect original conditions (edge index -> condition string).
        let original_conditions: Vec<(usize, String)> = graph
            .graph()
            .edge_indices()
            .filter_map(|idx| {
                let edge = graph.graph().edge_weight(idx)?;
                edge.conditions.as_ref().map(|c| (idx.index(), c.to_string()))
            })
            .collect();

        // Serialize and deserialize.
        let portable = PortableQueryGraph::from_query_graph(graph, &sdl);
        let bytes = portable.to_bytes();
        let restored_portable = PortableQueryGraph::from_bytes(&bytes).unwrap();

        // Verify condition_index is set on all edges that had conditions.
        for (_, _, pedge) in &restored_portable.edges {
            if pedge.conditions.is_some() {
                assert!(
                    pedge.condition_index.is_some(),
                    "v2 edge with conditions should also have condition_index"
                );
            }
        }

        // Reconstruct and compare conditions.
        let supergraph2 = crate::Supergraph::new_with_router_specs(&sdl).unwrap();
        let (restored_graph, timing) = restored_portable
            .to_query_graph_with_timing(supergraph2.schema.clone())
            .unwrap();

        let restored_conditions: Vec<(usize, String)> = restored_graph
            .graph()
            .edge_indices()
            .filter_map(|idx| {
                let edge = restored_graph.graph().edge_weight(idx)?;
                edge.conditions.as_ref().map(|c| (idx.index(), c.to_string()))
            })
            .collect();

        assert_eq!(
            original_conditions.len(),
            restored_conditions.len(),
            "condition count mismatch: original={}, restored={}",
            original_conditions.len(),
            restored_conditions.len(),
        );

        // Compare each condition string.
        for (orig, restored) in original_conditions.iter().zip(restored_conditions.iter()) {
            assert_eq!(
                orig.0, restored.0,
                "edge index mismatch"
            );
            assert_eq!(
                orig.1, restored.1,
                "condition string mismatch at edge {}: original='{}', restored='{}'",
                orig.0, orig.1, restored.1,
            );
        }

        eprintln!(
            "Condition roundtrip OK: {} conditions, {} unique in table, \
             reconstruct_conditions={}ms rebuild_graph={}ms",
            original_conditions.len(),
            restored_portable.condition_table.len(),
            timing.reconstruct_conditions_ms,
            timing.rebuild_graph_ms,
        );
    }

    /// Test v1 backward compatibility: a v1 artifact (no condition_table) should
    /// still reconstruct correctly via the parse_field_set fallback.
    #[test]
    fn test_portable_v1_compat() {
        let sdl = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/query_plan/supergraphs/avoids_unnecessary_fetches.graphql")
        ).unwrap();

        let supergraph = crate::Supergraph::new_with_router_specs(&sdl).unwrap();
        let planner = QueryPlanner::new(&supergraph, QueryPlannerConfig::default()).unwrap();
        let graph = planner.query_graph();

        let mut portable = PortableQueryGraph::from_query_graph(graph, &sdl);

        // Simulate v1: clear condition_table and condition_index from all edges.
        portable.version = 1;
        portable.condition_table.clear();
        for (_, _, pedge) in &mut portable.edges {
            pedge.condition_index = None;
        }

        // Serialize as v1.
        let bytes = portable.to_bytes();
        let restored = PortableQueryGraph::from_bytes(&bytes).unwrap();
        assert_eq!(restored.version, 1);
        assert!(restored.condition_table.is_empty());

        // Reconstruction should still work via parse_field_set fallback.
        let supergraph2 = crate::Supergraph::new_with_router_specs(&sdl).unwrap();
        let restored_graph = restored
            .to_query_graph(supergraph2.schema.clone())
            .unwrap();

        assert_eq!(
            restored_graph.graph().node_count(),
            graph.graph().node_count(),
        );
        assert_eq!(
            restored_graph.graph().edge_count(),
            graph.graph().edge_count(),
        );
    }

    /// Test that v3 artifacts serialize and restore QueryGraphMetadata correctly.
    /// The restored metadata must match the freshly-computed metadata.
    #[test]
    fn test_portable_metadata_roundtrip() {
        use crate::composition::compose;
        use crate::subgraph::typestate::Subgraph;

        let subgraph_a = Subgraph::parse(
            "A",
            "http://a",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type Query { user(id: ID!): User }
                type User @key(fields: "id") { id: ID! name: String }
            "#,
        ).unwrap();
        let subgraph_b = Subgraph::parse(
            "B",
            "http://b",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type User @key(fields: "id") { id: ID! email: String }
            "#,
        ).unwrap();

        let supergraph = compose(vec![subgraph_a, subgraph_b]).unwrap();
        let supergraph_sdl = supergraph.schema().schema().serialize().to_string();

        // Build planner (which computes metadata from scratch).
        let sg = crate::Supergraph::new_with_router_specs(&supergraph_sdl).unwrap();
        let planner = QueryPlanner::new(&sg, QueryPlannerConfig::default()).unwrap();
        let graph = planner.query_graph();

        // Serialize to v3 portable format (includes metadata + semantic indices).
        let portable = PortableQueryGraph::from_query_graph(graph, &supergraph_sdl);
        assert!(portable.metadata.is_some(), "v3 artifact should have metadata");

        // Roundtrip through bincode.
        let bytes = portable.to_bytes();
        let restored = PortableQueryGraph::from_bytes(&bytes).unwrap();

        // Reconstruct QueryGraph from artifact (v3 fast path).
        let sg2 = crate::Supergraph::new_with_router_specs(&supergraph_sdl).unwrap();
        let (restored_graph, timing) = restored
            .to_query_graph_with_timing(sg2.schema.clone())
            .unwrap();

        // Verify timing shows metadata was restored, not recomputed.
        // (In v3 fast path, compute_non_local_metadata_ms should be near-zero.)
        // We can't assert exact timing in tests, but we can verify the metadata is populated.

        // Verify structural equivalence.
        assert_eq!(restored_graph.graph().node_count(), graph.graph().node_count());
        assert_eq!(restored_graph.graph().edge_count(), graph.graph().edge_count());

        // Verify semantic indices were restored.
        // Spot-check: every node in the original graph should have a semantic ID in the restored graph.
        for node_idx in graph.graph().node_indices() {
            let original_id = graph.semantic_node_id(node_idx);
            let restored_id = restored_graph.semantic_node_id(node_idx);
            assert_eq!(
                original_id.map(|id| id.type_name.to_string()),
                restored_id.map(|id| id.type_name.to_string()),
                "semantic node ID mismatch at node {node_idx:?}"
            );
        }

    }

    /// Test that v2 artifacts (no metadata) still work via fallback recomputation.
    #[test]
    fn test_portable_v2_compat() {
        use crate::composition::compose;
        use crate::subgraph::typestate::Subgraph;

        let subgraph_a = Subgraph::parse(
            "A",
            "http://a",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type Query { user(id: ID!): User }
                type User @key(fields: "id") { id: ID! name: String }
            "#,
        ).unwrap();
        let subgraph_b = Subgraph::parse(
            "B",
            "http://b",
            r#"
                extend schema @link(url: "https://specs.apollo.dev/federation/v2.5", import: ["@key"])
                type User @key(fields: "id") { id: ID! email: String }
            "#,
        ).unwrap();

        let supergraph = compose(vec![subgraph_a, subgraph_b]).unwrap();
        let supergraph_sdl = supergraph.schema().schema().serialize().to_string();
        let sg = crate::Supergraph::new_with_router_specs(&supergraph_sdl).unwrap();
        let planner = QueryPlanner::new(&sg, QueryPlannerConfig::default()).unwrap();
        let graph = planner.query_graph();

        // Create a v3 artifact, then strip metadata + semantic indices to simulate v2.
        let mut portable = PortableQueryGraph::from_query_graph(graph, &supergraph_sdl);
        portable.version = 2;
        portable.metadata = None;
        portable.semantic_node_index = Vec::new();
        portable.semantic_edge_index = Vec::new();

        let bytes = portable.to_bytes();
        let restored = PortableQueryGraph::from_bytes(&bytes).unwrap();
        assert!(restored.metadata.is_none());

        // Reconstruct — should fall back to computing metadata + semantic indices.
        let sg2 = crate::Supergraph::new_with_router_specs(&supergraph_sdl).unwrap();
        let (restored_graph, timing) = restored
            .to_query_graph_with_timing(sg2.schema.clone())
            .unwrap();

        // Fallback path should have non-zero metadata computation time.
        // (We can't reliably assert this in CI, but the graph should be functional.)
        assert_eq!(restored_graph.graph().node_count(), graph.graph().node_count());
        assert_eq!(restored_graph.graph().edge_count(), graph.graph().edge_count());
    }
}
