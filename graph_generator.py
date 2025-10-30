# graph_generator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Protocol, Iterable
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData


# --------- Public “feature provider” interface (implemented elsewhere) ---------
class PropertyProvider(Protocol):
    """
    Interface that supplies property vectors (features) for each semantic instance and
    the overall object-level vector used for the MainObject node.

    Implement this in your graph_semantics_properties module.
    """

    def get_overall_features(self, obj_id: Any) -> torch.Tensor:
        """Return the overall property vector for the main object (1D tensor)."""

    def get_features(self, obj_id: Any, semantic_type: str, semantic_local_id: int) -> torch.Tensor:
        """
        Return the feature vector for a specific semantic instance, where
        `semantic_local_id` is the position/index in the CityJSON array for that type
        (preserve original order as given in the file).
        """


# --------- Configuration and constants ---------
MAIN_TYPE = "MainObject"

# Semantics you expect to see; order here doesn't affect node ordering in data (we preserve file order).
DEFAULT_TYPE_VOCAB = (
    "GroundSurface",
    "WallSurface",
    "RoofSurface",
    "Window",
    "Door",
)

# Which types the MainObject should connect to (bidirectionally)
MAIN_CONNECT_TYPES = frozenset({"GroundSurface", "WallSurface", "RoofSurface"})

# Relation names (you can rename them later if you prefer)
REL_PARENT_OF = "parent_of"
REL_CHILD_OF = "child_of"
REL_CONNECTS_TO = "connects_to"
REL_CONNECTED_FROM = "connected_from"


@dataclass(frozen=True)
class GraphGeneratorConfig:
    """
    Configuration knobs for GraphGenerator. Keep it minimal now; extend as needed.
    """
    type_vocab: Tuple[str, ...] = DEFAULT_TYPE_VOCAB
    preserve_cityjson_order: bool = True
    bidirectional_edges: bool = True
    connect_main_to_types: Iterable[str] = MAIN_CONNECT_TYPES


# --------- The GraphGenerator itself ---------
class GraphGenerator:
    """
    Build a per-object heterogeneous PyG graph from a CityJSON object dict.

    - One HeteroData per main object (single connected component).
    - Node types: semantic parts from CityJSON + a single MainObject node.
    - Edges:
        * From CityJSON semantics: parent_of / child_of across types.
        * MainObject <-> {GroundSurface, WallSurface, RoofSurface} (not to Door/Window).
    - Provenance:
        * Preserve exact per-type node ordering as in CityJSON (no re-sorting).
        * Store reversible maps from PyG indices back to original CityJSON positions.
    - Features:
        * Delegated to a PropertyProvider implementation (no feature math here).
    """

    def __init__(self, config: Optional[GraphGeneratorConfig] = None):
        self.cfg = config or GraphGeneratorConfig()

        # Basic sanity on vocab
        self._type_set = set(self.cfg.type_vocab)
        assert MAIN_TYPE not in self._type_set, "MAIN_TYPE must be separate from semantic vocab."

    # ---- Public API ----
    def build(self, cityjson_obj: Dict[str, Any], obj_id: Any, prop_provider: PropertyProvider) -> HeteroData:
        """
        Construct HeteroData for a single CityJSON object.

        Parameters
        ----------
        cityjson_obj : dict
            Parsed CityJSON dict for a single object (e.g., one building). Must contain
            a 'surfaces' array or equivalent semantic structure describing parts with
            'type', possibly 'parent' and 'children' fields that reference positions.
        obj_id : Any
            The original identifier of the main object in the source dataset.
        prop_provider : PropertyProvider
            An object that implements feature retrieval for each semantic instance
            and the overall main-object vector.

        Returns
        -------
        HeteroData
            A fully-formed hetero graph with x per node-type, typed edge_index tensors,
            and metadata for provenance/reversibility.
        """
        # 1) Collect semantic records and keep their original order
        sem_records = self._collect_semantics(cityjson_obj)

        # 2) Build per-type node lists (preserve order exactly as provided)
        node_lists = self._build_node_lists(sem_records)

        # 3) Build index/provenance mappings
        fwd_maps, rev_maps = self._build_mappings(node_lists)

        # 4) Attach features (x) per type (delegated to prop_provider)
        x_dict, feat_dims = self._attach_features(obj_id, node_lists, prop_provider)

        # 5) Build typed edges from semantics + MainObject connections
        edge_dict = self._build_edges(sem_records, fwd_maps)

        # 6) Finalize HeteroData with metadata
        data = self._finalize_heterodata(
            obj_id=obj_id,
            node_lists=node_lists,
            fwd_maps=fwd_maps,
            rev_maps=rev_maps,
            x_dict=x_dict,
            edge_dict=edge_dict,
            feat_dims=feat_dims,
        )
        return data

    # ---- Steps (kept simple & explicit; fill in logic in next iteration) ----
    def _collect_semantics(self, cityjson_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract the list of semantic records in the exact order they appear.

        Expected minimal schema per record:
            {
              "type": str,                  # e.g., "WallSurface"
              # Optional:
              "parent": int,                # index into this list (0-based)
              "children": List[int],        # indices into this list (0-based)
              # + any other fields you keep, e.g., vertices reference, etc.
            }

        Returns the raw list; do not sort or re-index.
        """
        surfaces = cityjson_obj.get("surfaces", [])
        # (Do not reorder; trust CityJSON to be already ordered.)
        return surfaces

    def _build_node_lists(self, sem_records: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Group semantic indices by type, preserving the original order.
        Returns:
            node_lists[type] = [list of indices into sem_records]
        """
        node_lists: Dict[str, List[int]] = {t: [] for t in self.cfg.type_vocab}
        for idx, rec in enumerate(sem_records):
            t = rec.get("type")
            if t not in self._type_set:
                # Unknown type: you can either skip or add dynamically. For now, add dynamically.
                node_lists.setdefault(t, [])
                self._type_set.add(t)
            node_lists[t].append(idx)
        return node_lists

    def _build_mappings(self, node_lists: Dict[str, List[int]]) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, int]]]:
        """
        Build forward and reverse maps:
            fwd_maps[type][orig_idx] = local_idx_within_type
            rev_maps[type][local_idx] = orig_idx

        Here, orig_idx is the position in `sem_records` (which is the CityJSON order).
        """
        fwd_maps: Dict[str, Dict[int, int]] = {}
        rev_maps: Dict[str, Dict[int, int]] = {}
        for t, orig_indices in node_lists.items():
            fwd = {orig_idx: local_i for local_i, orig_idx in enumerate(orig_indices)}
            rev = {local_i: orig_idx for local_i, orig_idx in enumerate(orig_indices)}
            fwd_maps[t] = fwd
            rev_maps[t] = rev
        return fwd_maps, rev_maps

    def _attach_features(
        self,
        obj_id: Any,
        node_lists: Dict[str, List[int]],
        prop_provider: PropertyProvider
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        """
        Ask the prop_provider for features, preserving ordering.

        Returns:
            x_dict[type] = Tensor[num_nodes_type, feat_dim_type]
            feat_dims[type] = int
        Also includes the MAIN_TYPE with shape [1, D_main].
        """
        x_dict: Dict[str, torch.Tensor] = {}
        feat_dims: Dict[str, int] = {}

        # Per-type semantic features
        for t, orig_indices in node_lists.items():
            if len(orig_indices) == 0:
                # Create an empty (0, ?) placeholder tensor later at model time if needed.
                # For now, store a zero-node placeholder to keep schema explicit.
                x_dict[t] = torch.empty((0, 0), dtype=torch.float32)
                feat_dims[t] = 0
                continue

            feats: List[torch.Tensor] = []
            for local_i, orig_idx in enumerate(orig_indices):
                # local_i is just iteration order; orig_idx is the CityJSON position we must preserve.
                f = prop_provider.get_features(obj_id=obj_id, semantic_type=t, semantic_local_id=orig_idx)
                assert f.dim() == 1, "Feature vector must be 1D (no batching here)."
                feats.append(f)

            x_t = torch.stack(feats, dim=0) if feats else torch.empty((0, 0), dtype=torch.float32)
            x_dict[t] = x_t
            feat_dims[t] = 0 if x_t.numel() == 0 else x_t.size(-1)

        # MainObject features from overall vector
        x_main = prop_provider.get_overall_features(obj_id)
        assert x_main.dim() == 1, "MainObject overall feature must be 1D."
        x_dict[MAIN_TYPE] = x_main.view(1, -1)
        feat_dims[MAIN_TYPE] = x_main.numel()

        return x_dict, feat_dims

    def _build_edges(
        self,
        sem_records: List[Dict[str, Any]],
        fwd_maps: Dict[str, Dict[int, int]],
    ) -> Dict[Tuple[str, str, str], List[Tuple[int, int]]]:
        """
        Create typed edges from:
          1) Semantics parent/children fields (both directions if configured).
          2) MainObject <-> {GroundSurface, WallSurface, RoofSurface}.

        Returns:
            edge_dict[(src_type, rel, dst_type)] = list of (src_idx, dst_idx)
        """
        edge_dict: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)

        # 1) From semantics
        for child_idx, rec in enumerate(sem_records):
            child_type = rec.get("type")
            parent_idx = rec.get("parent", None)

            if parent_idx is not None:
                # Parent -> child
                parent_type = sem_records[parent_idx].get("type")
                src_i = fwd_maps[parent_type][parent_idx]
                dst_i = fwd_maps[child_type][child_idx]
                edge_dict[(parent_type, REL_PARENT_OF, child_type)].append((src_i, dst_i))

                # Child -> parent
                if self.cfg.bidirectional_edges:
                    edge_dict[(child_type, REL_CHILD_OF, parent_type)].append((dst_i, src_i))

            # If "children" is present, it should be consistent with parent; we don't need to duplicate.

        # 2) MainObject connections (to configured types only)
        for t in self.cfg.connect_main_to_types:
            if t not in fwd_maps:
                continue
            for orig_idx in fwd_maps[t].keys():
                local_i = fwd_maps[t][orig_idx]
                edge_dict[(MAIN_TYPE, REL_CONNECTS_TO, t)].append((0, local_i))
                if self.cfg.bidirectional_edges:
                    edge_dict[(t, REL_CONNECTED_FROM, MAIN_TYPE)].append((local_i, 0))

        return edge_dict

    def _finalize_heterodata(
        self,
        obj_id: Any,
        node_lists: Dict[str, List[int]],
        fwd_maps: Dict[str, Dict[int, int]],
        rev_maps: Dict[str, Dict[int, int]],
        x_dict: Dict[str, torch.Tensor],
        edge_dict: Dict[Tuple[str, str, str], List[Tuple[int, int]]],
        feat_dims: Dict[str, int],
    ) -> HeteroData:
        """
        Convert everything to a HeteroData instance and attach provenance metadata.
        """
        data = HeteroData()

        # Node stores
        for t, x in x_dict.items():
            data[t].x = x  # shape [num_nodes_t, feat_dim_t]

        # Edge stores
        for (src_t, rel, dst_t), pairs in edge_dict.items():
            if len(pairs) == 0:
                # Create an empty edge_index with correct shape
                data[(src_t, rel, dst_t)].edge_index = torch.empty((2, 0), dtype=torch.long)
                continue
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()  # [2, E]
            data[(src_t, rel, dst_t)].edge_index = edge_index

        # Metadata for reversibility / debugging
        data._meta = {
            "obj_id": obj_id,
            "main_type": MAIN_TYPE,
            "feat_dims": feat_dims,
            "node_registry": {t: {"ids_in_cityjson_order": node_lists[t],
                                  "orig_to_local": fwd_maps[t],
                                  "local_to_orig": rev_maps[t]}
                              for t in node_lists.keys()},
            "version": "graph_generator:v1",
        }
        # Quick access to obj_id on the main store as well
        data[MAIN_TYPE].obj_id = obj_id

        return data
