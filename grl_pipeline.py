from utils import *
from collections import defaultdict
import random
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List, Tuple, Optional, DefaultDict
from feature_generators import TYPE_GENERATORS
from data_preparation.grl_pair_dataset import PairDataset

from torch.utils.data import DataLoader
from training import Trainer, Evaluator, compute_metrics
from torch_geometric.data import Batch
from models import GraphEncoder, PairMatcher


class GRLPipelineManager:
    def __init__(self, seed, logger, args):
        self.seed = seed
        self.logger = logger
        self.dataset_name = args.dataset_name
        self.load_cached_graphs = args.load_cached_graphs
        self.load_cached_pairs = args.load_cached_pairs
        self.suffix = args.suffix
        self.min_surfaces_num = args.min_surfaces_num
        self.neg_pairs_num = args.neg_pairs_num
        self.pair_aggregation = args.pair_aggregation
        self.result_dict = defaultdict(dict)
        self.type_generators = TYPE_GENERATORS
        self.training_epochs = args.training_epochs
        self.training_params = self._get_training_params(args)
        self.feature_config = self._load_json("feature_generators/type_features_dict.json")
        self.graph_objects = self._get_or_load_graph_objects()
        self.pairs = self._get_pairs()
        self.datasets = self._build_or_load_datasets()
        self.classifier, self.eval_artifacts = self._train_and_evaluate()

    def _get_or_load_graph_objects(self):
        cache_path = f"data/graph_objects/{self.dataset_name}/graph_objects_seed{self.seed}_{self.suffix}.pkl"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if self.load_cached_graphs and os.path.exists(cache_path):
            self.logger.info(f"Loading cached graph_objects from {cache_path}")
            with open(cache_path, "rb") as f:
                graph_objects = pkl.load(f)
            self._add_relations(graph_objects)
        else:
            self.logger.info("Building graph_objects from scratch...")
            graph_objects = self._build_graphs_wrapper()
            self._add_relations(graph_objects)
            with open(cache_path, "wb") as f:
                pkl.dump(graph_objects, f)
        return graph_objects

    @staticmethod
    def _get_training_params(args):
        return {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "gnn_layers_num": args.gnn_layers_num,
            "batch_size": args.batch_size,
            "hidden_dim1": args.hidden_dim1,
            "hidden_dim2": args.hidden_dim2,
            "hidden_dim3": args.hidden_dim3,
            "dropout_rate": args.dropout_rate,
        }

    def _add_relations(self, graph_objects):
        all_relations = set()
        for subset_dict in graph_objects.values():
            for g in subset_dict.values():
                all_relations.update(g["edges"].keys())
        self.feature_config["relations"] = list(all_relations)

    def _build_graphs_wrapper(self):
        dataset_config = json.load(open('dataset_configs.json'))[self.dataset_name]
        graph_dict = {'cands':  self._extract_raw_graphs(dataset_config['cands_path']),
                      'index': self._extract_raw_graphs(dataset_config['index_path'])}
        return graph_dict

    def _extract_raw_graphs(self, path: str):
        """
        Parallel version: process all CityJSON objects with multiprocessing.
        """
        raw = self._load_json(path)
        obj_items = list(self._iter_objects(raw, path))
        self.logger.info(f"Processing {len(obj_items)} objects with multiprocessing...")

        args = [
            (obj_id, obj, self.min_surfaces_num, self.type_generators, self.feature_config, self.logger)
            for obj_id, obj in obj_items
        ]

        graphs: Dict[Any, Dict[str, Any]] = {}
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            for res in pool.starmap(GRLPipelineManager._process_single_object, args):
                if res is None:
                    continue
                obj_id, graph = res
                graphs[obj_id] = graph

        return graphs

    @staticmethod
    def _process_single_object(
            obj_id: str,
            obj: Dict[str, Any],
            min_surfaces_num: int,
            type_generators: Dict[str, Any],
            feature_config: Dict[str, Any],
            logger: Optional[Any] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Static helper: process a single CityJSON object into a graph dict.
        Skips objects without enough surfaces or without a valid MainObject vector.
        """
        try:
            surfaces = obj.get("geometry", [])[0].get("semantics", {}).get("surfaces", [])
            if len(surfaces) < min_surfaces_num:
                return None
            node_lists = GRLPipelineManager._collect_node_lists(surfaces)
            fwd_maps, rev_maps = GRLPipelineManager._build_type_maps(node_lists)
            edges = GRLPipelineManager._build_edges(
                surfaces=surfaces,
                forward_maps=fwd_maps,
                main_connection_types=("GroundSurface", "WallSurface", "RoofSurface"),
                bidirectional=True,
            )
            x_by_type, feat_dims = GRLPipelineManager._generate_surface_features(
                node_lists, obj, type_generators, feature_config, logger
            )
            x_main, main_dim = GRLPipelineManager._generate_mainobject_vector(
                obj, type_generators, feature_config, logger
            )
            if x_main is None:
                return None
            feat_dims["MainObject"] = main_dim
            graph = GRLPipelineManager._assemble_graph(
                obj_id, node_lists, fwd_maps, rev_maps, edges, x_by_type, x_main, feat_dims
            )
            return obj_id, graph

        except Exception:
            return None

    @staticmethod
    def _generate_surface_features(node_lists, obj, type_generators, feature_config, logger):
        """
        Generate feature tensors for all semantic surface types.
        Apply log scaling to compress large ranges.
        Returns: (x_by_type: Dict[str, np.ndarray], feat_dims: Dict[str, int])
        """
        x_by_type, feat_dims = {}, {}
        for semantic_type, original_indices in node_lists.items():
            if semantic_type == "MainObject":
                continue
            gen_cls = type_generators.get(semantic_type)
            if gen_cls is None or not original_indices:
                x_by_type[semantic_type] = torch.empty((0, 0), dtype=torch.float32)
                feat_dims[semantic_type] = 0
                continue
            feature_spec = feature_config.get(semantic_type, [])
            generator = gen_cls(obj=obj, feature_spec=feature_spec, logger=logger)
            expected_dim = generator.dim()
            features_1d = [
                GRLPipelineManager._ensure_1d_tensor(generator.get_feature_vector(i), expected_dim)
                for i in original_indices
            ]
            x_t = torch.stack(features_1d, dim=0) if features_1d else torch.empty((0, 0), dtype=torch.float32)
            # if x_t.numel() > 0:
            #     x_t = torch.clamp(x_t, min=0)
            #     x_t = torch.log1p(x_t)
            x_by_type[semantic_type] = x_t
            feat_dims[semantic_type] = 0 if x_t.numel() == 0 else x_t.size(-1)
        x_by_type = {k: v.cpu().numpy() for k, v in x_by_type.items()}
        return x_by_type, feat_dims

    @staticmethod
    def _generate_mainobject_vector(obj, type_generators, feature_config, logger):
        """
        Generate the MainObject vector. Returns (x_main: np.ndarray | None, dim: int).
        Apply log scaling to compress large ranges.
        Returns None if invalid.
        """
        main_spec = feature_config.get("MainObject", [])
        main_cls = type_generators.get("MainObject")
        if main_cls is None:
            return None, 0
        gen = main_cls(obj=obj, feature_spec=main_spec, logger=logger)
        x_main = gen.get_vector()
        if not isinstance(x_main, torch.Tensor):
            x_main = torch.tensor(x_main, dtype=torch.float32).flatten()
        else:
            x_main = x_main.flatten().to(torch.float32)
        x_main = torch.where(torch.isfinite(x_main), x_main, torch.zeros_like(x_main))
        if x_main.numel() == 0:
            return None, 0
        x_main = x_main.view(1, -1)
        # x_main = torch.clamp(x_main, min=0)
        # x_main = torch.log1p(x_main)
        return x_main.cpu().numpy(), x_main.size(-1)

    @staticmethod
    def _assemble_graph(obj_id, node_lists, fwd_maps, rev_maps, edges, x_by_type, x_main, feat_dims):
        """
        Assemble final graph dict in numpy form.
        """
        return {
            "obj_id": obj_id,
            "node_lists": node_lists,
            "orig_to_type_local": fwd_maps,
            "type_local_to_orig": rev_maps,
            "edges": edges,
            "x_by_type": x_by_type,
            "x_main": x_main,
            "feat_dims": feat_dims,
        }

    def _train_and_evaluate(self):
        train_loader = self._get_data_loader("train")
        val_loader = self._get_data_loader("validation")
        test_loader = self._get_data_loader("test")
        encoder = GraphEncoder(self.feature_config, self.training_params)
        classifier = PairMatcher(
            encoder,
            aggregation=self.pair_aggregation,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        # criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(classifier.parameters(),
                                      lr=self.training_params["lr"],
                                      weight_decay=self.training_params["weight_decay"])
        trainer = Trainer(
            model=classifier,
            optimizer=optimizer,
            criterion=criterion,
            logger=self.logger,
        )
        trainer.fit(train_loader, val_loader, num_epochs=self.training_epochs, monitor="f1")
        eval_metrics, eval_preds = trainer.evaluate(test_loader, return_preds=True)
        eval_artifacts = {
            "metrics": eval_metrics,
            "predictions": eval_preds["preds"],
            "labels": eval_preds["labels"],
        }
        self._save_model(classifier, optimizer, eval_artifacts)
        return classifier, eval_artifacts

    @staticmethod
    def _collate_pair_batch(batch):
        """
        Collate function for DataLoader that batches pairs of HeteroData.
        """
        g1_list, g2_list, labels = zip(*batch)
        batched_g1 = Batch.from_data_list(g1_list)
        batched_g2 = Batch.from_data_list(g2_list)
        labels = torch.stack(labels)
        return batched_g1, batched_g2, labels

    def _get_data_loader(self, split: str, batch_size: int = 16, shuffle: bool = True):
        dataset = self.datasets[split]
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train" and shuffle),
            collate_fn=self._collate_pair_batch,
        )

    def _save_model(self, classifier, optimizer, eval_artifacts):
        save_dir = os.path.join("experiments", "runs", self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"classifier_and_run_artifacts_seed{self.seed}_{self.suffix}.pt")

        checkpoint = {
            "model_state": classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "dataset": self.dataset_name,
                "suffix": self.suffix,
                "aggregation": self.pair_aggregation,
            },
            "eval_metrics": eval_artifacts["metrics"],
            "predictions": eval_artifacts["predictions"],  # numpy array
            "labels": eval_artifacts["labels"],  # numpy array
        }
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved trained model to {save_path}")

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _collect_vertex_indices(boundaries: list) -> set:
        """Recursively collect all vertex indices from nested boundary lists."""
        indices = set()

        def recurse(boundary):
            if isinstance(boundary, int):
                indices.add(boundary)
            elif isinstance(boundary, list):
                for b in boundary:
                    recurse(b)
        recurse(boundaries)
        return indices

    def _iter_objects(self, raw: dict, path: str):
        """
        Yield (obj_id, obj) from a multi-object CityJSON container.
        - Attaches local vertex list + mappings for each object.
        """
        assert "CityObjects" in raw and isinstance(raw["CityObjects"], dict), \
            f"{path}: expected raw['CityObjects'] as a dict"
        assert "vertices" in raw, f"{path}: expected root-level 'vertices'"
        global_vertices = raw["vertices"]
        for obj_id, obj in raw["CityObjects"].items():
            # Collect all global vertex indices used in this object
            used_indices = set()
            for geom in obj.get("geometry", []):
                boundaries = geom.get("boundaries", [])
                used_indices.update(self._collect_vertex_indices(boundaries))
            sorted_indices = sorted(used_indices)
            global_to_local = {g: l for l, g in enumerate(sorted_indices)}
            local_to_global = {l: g for g, l in global_to_local.items()}
            obj = {
                **obj,
                "vertices": [global_vertices[g] for g in sorted_indices],
                "global_to_local": global_to_local,
                "local_to_global": local_to_global,
            }
            yield obj_id, obj

    @staticmethod
    def _collect_node_lists(surfaces: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Group original indices by semantic type, preserving order exactly as in `surfaces`.
        node_lists[type] = [orig_idx, ...]
        """
        node_lists: Dict[str, List[int]] = defaultdict(list)
        node_lists["MainObject"] = [len(surfaces)]
        for i, rec in enumerate(surfaces):
            node_lists[rec["type"]].append(i)
        return dict(node_lists)

    @staticmethod
    def _build_type_maps(
            node_lists: Dict[str, List[int]]
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, int]]]:
        """
        Build per-type forward and reverse maps between original surface positions and per-type local indices.
        
        Args:
            node_lists: {semantic_type: [orig_idx, ...]}

        Returns:
            fwd_maps_dict: {semantic_type: {orig_idx: local_idx}}
            rev_maps_dict: {semantic_type: {local_idx: orig_idx}}
            
        """
        fwd_maps_dict: Dict[str, Dict[int, int]] = {}
        rev_maps_dict: Dict[str, Dict[int, int]] = {}
        for semantic_type, orig_indices in node_lists.items():
            fwd_maps_dict[semantic_type] = {orig_idx: j for j, orig_idx in enumerate(orig_indices)}
            rev_maps_dict[semantic_type] = {j: orig_idx for j, orig_idx in enumerate(orig_indices)}
        return fwd_maps_dict, rev_maps_dict

    @staticmethod
    def _build_edges(
            surfaces: List[Dict[str, Any]],
            forward_maps: Dict[str, Dict[int, int]],
            main_connection_types: Tuple[str, ...],
            bidirectional: bool = True
    ) -> Dict[Tuple[str, str, str], List[Tuple[int, int]]]:
        """
        Orchestrator: build typed edge lists using two focused helpers.
        """
        edge_lists: DefaultDict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
        GRLPipelineManager._add_parent_child_edges(surfaces, forward_maps, edge_lists, bidirectional)
        GRLPipelineManager._add_mainobject_edges(forward_maps, main_connection_types, edge_lists, bidirectional)
        return dict(edge_lists)

    @staticmethod
    def _add_parent_child_edges(
            surfaces: List[Dict[str, Any]],
            forward_maps: Dict[str, Dict[int, int]],
            edge_lists: DefaultDict[Tuple[str, str, str],
            List[Tuple[int, int]]],
            bidirectional: bool
    ) -> None:
        """
        Add parent_of / child_of edges derived from CityJSON semantics.
        Uses per-type local indices from `forward_maps`.
        """
        for child_orig_index, child_record in enumerate(surfaces):
            parent_orig_index = child_record.get("parent")
            if parent_orig_index is None:
                continue

            child_type = child_record["type"]
            parent_type = surfaces[parent_orig_index]["type"]

            parent_local = forward_maps[parent_type][parent_orig_index]
            child_local = forward_maps[child_type][child_orig_index]

            edge_lists[(parent_type, "parent_of", child_type)].append((parent_local, child_local))
            if bidirectional:
                edge_lists[(child_type, "child_of", parent_type)].append((child_local, parent_local))
        return

    @staticmethod
    def _add_mainobject_edges(
            forward_maps: Dict[str, Dict[int, int]],
            main_connection_types: Tuple[str, ...],
            edge_lists: DefaultDict[Tuple[str, str, str], List[Tuple[int, int]]],
            bidirectional: bool,
    ) -> None:
        """
        Connect MainObject (unique node, index = len(surfaces)) to selected semantic types
        (e.g., GroundSurface, WallSurface, RoofSurface).
        """
        # Retrieve the local index of the MainObject
        if "MainObject" not in forward_maps:
            return
        main_local = list(forward_maps["MainObject"].values())[0]

        for semantic_type in main_connection_types:
            local_map = forward_maps.get(semantic_type)
            if not local_map:
                continue
            for _orig_idx, local_idx in local_map.items():
                edge_lists[("MainObject", "connects_to", semantic_type)].append((main_local, local_idx))
                if bidirectional:
                    edge_lists[(semantic_type, "connected_from", "MainObject")].append((local_idx, main_local))

    def _attach_features(
            self,
            obj_id: Any,
            node_lists: Dict[str, List[int]],
            obj: Dict[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, int]]:
        """
        Build per-type feature matrices + MainObject overall vector, by calling
        per-type feature generators.

        Args:
            obj_id: identifier of the current CityJSON object
            node_lists: {semantic_type: [orig_idx, ...]}
            obj: full CityJSON object dict (with 'surfaces' and 'vertices')

        Expectations:
          - self.type_generators: Dict[str, Type[BaseFeaturesGenerator]]
            e.g., {"Window": WindowFeaturesGenerator, ...}
          - self.main_vector_source: has `vector_for(obj_id) -> 1D torch.Tensor`
          - (optional) self.features_config: Dict[str, List[SpecItem]]
        """
        generator_cache = {}
        x_by_type, feat_dims = {}, {}

        for semantic_type, original_indices in node_lists.items():
            generator = self._get_or_create_feature_vec(semantic_type, obj, generator_cache)
            if generator is None or not original_indices:
                x_by_type[semantic_type] = torch.empty((0, 0), dtype=torch.float32)
                feat_dims[semantic_type] = 0
                continue

            expected_dim = generator.dim()  # <- one source of truth for this type
            features_1d = [
                self._ensure_1d_tensor(generator.get_feature_vector(i), expected_dim)
                for i in original_indices
            ]

            x_t = torch.stack(features_1d, dim=0) if features_1d else torch.empty((0, 0), dtype=torch.float32)
            x_by_type[semantic_type] = x_t
            feat_dims[semantic_type] = 0 if x_t.numel() == 0 else x_t.size(-1)

        x_main_1d = self._get_overall_vector(obj_id, obj)
        x_main = x_main_1d.view(1, -1) if x_main_1d.numel() > 0 else torch.empty((1, 0), dtype=torch.float32)
        feat_dims["MainObject"] = x_main.size(-1)
        return x_by_type, x_main, feat_dims

    def _get_or_create_feature_vec(
            self,
            semantic_type: str,
            obj: Dict[str, Any],
            cache: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Return a cached per-type feature generator for this object, creating it once if registered.
        """
        if semantic_type in cache:
            return cache[semantic_type]

        gen_cls = self.type_generators.get(semantic_type)
        if gen_cls is None:
            if getattr(self, "logger", None):
                self.logger.debug(f"No generator registered for semantic type '{semantic_type}'.")
            return None
        try:
            feature_spec = self.feature_config[semantic_type]
            cache[semantic_type] = gen_cls(obj=obj, feature_spec=feature_spec, logger=getattr(self, "logger", None))
            return cache[semantic_type]
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.warning(f"Failed to init generator for type '{semantic_type}': {e}")
            return None

    @staticmethod
    def _ensure_1d_tensor(vec, target_dim: int) -> torch.Tensor:
        """
        Make sure `vec` is a 1D float32 torch tensor of length `target_dim`.
        - Flattens inputs
        - Pads with zeros or truncates as needed
        - Replaces non-finite values with 0
        """
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
        else:
            vec = vec.to(dtype=torch.float32).flatten()

        # sanitize non-finite
        if not torch.isfinite(vec).all():
            vec = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))

        n = vec.numel()
        if n == target_dim:
            return vec
        if n > target_dim:
            return vec[:target_dim]

        out = torch.zeros(target_dim, dtype=torch.float32)
        if n > 0:
            out[:n] = vec
        return out

    def _get_overall_vector(self, obj_id: Any, obj: Dict[str, Any]) -> torch.Tensor:
        """
        Build the MainObject vector directly from `obj` using MainObjectFeaturesGenerator.
        Uses config if present under key "MainObject"; otherwise the class DEFAULT_SPEC.
        """
        gen_cls = self.type_generators.get("MainObject")
        if gen_cls is None:
            return torch.empty(0, dtype=torch.float32)

        spec_cfg = getattr(self, "features_config", {}) or {}
        main_spec = spec_cfg.get("MainObject")  # may be None → class default
        # try:
        gen = gen_cls(obj=obj, feature_spec=main_spec, logger=getattr(self, "logger", None))
        vec = gen.get_vector()  # whole-object vector
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32).flatten()
        else:
            vec = vec.flatten().to(torch.float32)
        vec = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
        return vec
        # except Exception as e:
        #     if getattr(self, "logger", None):
        #         self.logger.warning(f"Failed to compute MainObject vector for {obj_id}: {e}")
        #     return torch.empty(0, dtype=torch.float32)

    def _get_pairs(self, val_ratio: float = 0.2) -> Dict[str, List[Tuple[str, str, int]]]:
        """
        Load candidate pairs from dataset_partition_dict.
        Train is split into train/validation.
        Test is loaded directly.
        Labels are 1 if id1 == id2, else 0.
        Invalid pairs (objects missing from graph_objects) are skipped.
        """
        if self.load_cached_pairs:
            return None
        dataset_partition_dict = self._load_partition_dict()
        valid_ids = set(self.graph_objects["cands"].keys()) | set(self.graph_objects["index"].keys())
        train_labeled = self._label_pairs(
            dataset_partition_dict["train"]["blocking-based"]['large'][self.neg_pairs_num]
        )
        rng = random.Random(self.seed)
        rng.shuffle(train_labeled)
        train_pairs, val_pairs = self._split_train_val(train_labeled, val_ratio)
        test_labeled = self._label_pairs(dataset_partition_dict["test"]['matching']["blocking-based"]['large'][self.neg_pairs_num])
        pairs = {
            "train": self._filter_valid(train_pairs, valid_ids),
            "validation": self._filter_valid(val_pairs, valid_ids),
            "test": self._filter_valid(test_labeled, valid_ids),
        }
        self._log_pair_counts(pairs)
        return pairs

    def _load_partition_dict(self) -> Dict[str, Any]:
        dir_path = "data/dataset_partitions/"
        fname = f"{self.dataset_name}_seed{self.seed}.pkl"
        full_path = os.path.join(dir_path, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Partition file not found: {full_path}")
        with open(full_path, "rb") as f:
            return pkl.load(f)

    @staticmethod
    def _label_pairs(raw_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, int]]:
        """Convert raw (id1,id2) pairs into (id1,id2,label)."""
        return [(id1, id2, 1 if id1 == id2 else 0) for id1, id2 in raw_pairs]

    @staticmethod
    def _split_train_val(pairs: List[Tuple[str, str, int]], val_ratio: float):
        """Split labeled pairs into train/val subsets."""
        split_idx = int(len(pairs) * (1 - val_ratio))
        return pairs[:split_idx], pairs[split_idx:]

    @staticmethod
    def _filter_valid(pairs: List[Tuple[str, str, int]], valid_ids: set):
        """Remove pairs that reference objects not present in graph_objects."""
        return [(id1, id2, label) for id1, id2, label in pairs
                if id1 in valid_ids and id2 in valid_ids]

    def _log_pair_counts(self, pairs: Dict[str, List[Tuple[str, str, int]]]):
        self.logger.info(
            f"Pairs loaded successfully (after filtering invalid IDs). "
            f"Train: {len(pairs['train'])}, "
            f"Validation: {len(pairs['validation'])}, "
            f"Test: {len(pairs['test'])}"
        )

    def _build_or_load_datasets(self) -> Dict[str, PairDataset]:
        cache_dir = f"data/cache/{self.dataset_name}/"
        os.makedirs(cache_dir, exist_ok=True)
        datasets = {}

        for split in ["train", "validation", "test"]:
            save_path = os.path.join(cache_dir, f"{split}_pairs.pkl")

            if self.load_cached_pairs and os.path.exists(save_path):
                self.logger.info(f"Loading cached PairDataset for {split} from {save_path}")
                datasets[split] = PairDataset.load(save_path)
            else:
                self.logger.info(f"Building PairDataset for {split} and saving to {save_path}")
                datasets[split] = PairDataset(
                    self.pairs[split],
                    self.graph_objects,
                    save_path=save_path,
                )
        return datasets

