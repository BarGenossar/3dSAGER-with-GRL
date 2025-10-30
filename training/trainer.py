from training.evaluator import Evaluator
import torch

class Trainer:
    def __init__(self, model, optimizer, criterion, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger

    def fit(self, train_loader, val_loader, num_epochs: int, monitor: str = "f1"):
        best_state, best_val, best_metrics = None, -float("inf"), None
        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            self._log_epoch(epoch, train_loss, val_metrics)

            best_state, best_val, best_metrics = self._update_best(
                val_metrics, best_state, best_val, best_metrics, monitor, epoch
            )

        if best_state:
            self.model.load_state_dict(best_state["model"])
            self.logger.info(
                f"Restored best model from epoch {best_state['epoch']} "
                f"with {monitor}={best_val:.3f}"
            )
        return best_metrics

    def _train_one_epoch(self, loader):
        self.model.train()
        total_loss, total_samples = 0.0, 0
        for g1, g2, labels in loader:
            g1, g2, labels = g1.to(self.device), g2.to(self.device), labels.to(self.device).long()
            self.optimizer.zero_grad()
            logits = self.model(g1, g2)
            # loss = self.criterion(logits, labels)
            # loss = self.criterion(logits.view(-1), labels)
            loss = self.criterion(logits, labels.float())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        return total_loss / total_samples

    def _validate(self, loader):
        evaluator = Evaluator(self.model, device=self.device)
        return evaluator.run(loader)

    def _log_epoch(self, epoch, train_loss, val_metrics):
        self.logger.info(
            f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Val metrics: {val_metrics}"
        )

    def _update_best(self, val_metrics, best_state, best_val, best_metrics, monitor, epoch):
        metric_val = val_metrics.get(monitor)
        if metric_val is None:
            raise ValueError(f"Monitor metric {monitor} not found in {val_metrics}")

        if metric_val > best_val:
            best_val = metric_val
            best_state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": val_metrics,
                "epoch": epoch,
            }
            best_metrics = val_metrics
        return best_state, best_val, best_metrics

    def evaluate(self, loader, return_preds: bool = False):
        evaluator = Evaluator(self.model, device=self.device)
        return evaluator.run(loader, return_preds=return_preds)
