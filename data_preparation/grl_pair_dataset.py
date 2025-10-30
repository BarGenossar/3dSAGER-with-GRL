import torch
from torch.utils.data import Dataset
import pickle as pkl
import os
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData


class PairDataset(Dataset):
    """
    PyTorch Dataset for pairwise graph classification.
    Each item is (HeteroData1, HeteroData2, label).
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        graph_objects: Dict[str, Dict],
        save_path: str = None,
    ):
        """
        Args:
            pairs (list): list of (obj_id1, obj_id2, label).
            graph_objects (dict): {"cands": {id->graph}, "index": {id->graph}}.
            save_path (str, optional): where to save the dataset (pickle).
        """
        self.pairs = pairs
        self.graph_objects = graph_objects
        self.save_path = save_path

        if save_path is not None:
            self._save()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        obj1, obj2, label = self.pairs[idx]
        g1 = self._to_heterodata(self._resolve_graph(obj1))
        g2 = self._to_heterodata(self._resolve_graph(obj2))
        return g1, g2, torch.tensor(label, dtype=torch.float32)

    def _resolve_graph(self, obj_id: str) -> Dict:
        """
        Search both 'cands' and 'index' subdicts for obj_id.
        """
        for subset in ["cands", "index"]:
            if obj_id in self.graph_objects.get(subset, {}):
                return self.graph_objects[subset][obj_id]
        raise KeyError(f"Object ID {obj_id} not found in graph_objects")

    def _to_heterodata(self, graph_dict: Dict) -> HeteroData:
        """
        Convert one graph dict into a PyTorch Geometric HeteroData object.
        """
        data = HeteroData()
        for node_type, x in graph_dict["x_by_type"].items():
            x = torch.as_tensor(x, dtype=torch.float32)
            if x.numel() > 0:
                data[node_type].x = x
        x_main = torch.as_tensor(graph_dict["x_main"], dtype=torch.float32)
        if x_main.numel() > 0:
            data["MainObject"].x = x_main
        for (src_t, rel, dst_t), edge_list in graph_dict["edges"].items():
            if not edge_list:
                continue
            src, dst = zip(*edge_list)
            data[(src_t, rel, dst_t)].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )
        return data

    def _save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def load(path: str) -> "PairDataset":
        with open(path, "rb") as f:
            return pkl.load(f)

