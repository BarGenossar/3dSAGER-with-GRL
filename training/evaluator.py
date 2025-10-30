import torch
from training.metrics import compute_metrics


class Evaluator:
    def __init__(self, model, device="cpu", pred_threshold=None):
        self.model = model.to(device)
        self.device = device
        self.pred_threshold = 0.5 if pred_threshold is None else pred_threshold

    def run(self, loader, return_preds: bool = False):
        """
        Evaluate the model on a given dataloader.
        Args:
            loader (DataLoader): dataloader with (g1, g2, labels)
            return_preds (bool): if True, also return predictions + labels

        Returns:
            dict: metrics (always)
            dict (optional): {"preds": preds, "labels": labels_all}
        """
        self.model.eval()
        preds, labels_all = [], []

        with torch.no_grad():
            for g1, g2, labels in loader:
                g1, g2 = g1.to(self.device), g2.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(g1, g2)
                probs = torch.sigmoid(logits).cpu()

                preds.append(probs)
                labels_all.append(labels.cpu())

        preds = torch.cat(preds).numpy()
        labels_all = torch.cat(labels_all).numpy()
        # preds_class = preds.argmax(axis=1)
        preds_class = (preds >= self.pred_threshold).astype(int)
        metrics = compute_metrics(preds_class, labels_all)

        if return_preds:
            return metrics, {"preds": preds, "labels": labels_all}
        return metrics
