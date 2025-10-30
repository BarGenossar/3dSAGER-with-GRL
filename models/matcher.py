import torch
import torch.nn as nn


class PairMatcher(nn.Module):
    def __init__(self, encoder, hidden_dim=128, aggregation="concat"):
        """
        Match two graph embeddings.

        Args:
            encoder (nn.Module): Graph encoder producing embeddings.
            hidden_dim (int): hidden size for the MLP.
            aggregation (str): method to combine embeddings:
                - "concat": concatenate h1 and h2
                - "abs_diff": absolute difference |h1 - h2|
                - "division": element-wise division h1 / h2
                - "all": concat [h1, h2, |h1 - h2|]
        """
        super().__init__()
        self.encoder = encoder
        self.aggregation = aggregation.lower()

        if self.aggregation == "concat":
            in_dim = encoder.out_dim * 2
        elif self.aggregation == "abs_diff":
            in_dim = encoder.out_dim
        elif self.aggregation == "division":
            in_dim = encoder.out_dim
        elif self.aggregation == "all":
            in_dim = encoder.out_dim * 3
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # two classes instead of one
        )

    def forward(self, g1, g2):
        h1 = self.encoder(g1)
        h2 = self.encoder(g2)

        if self.aggregation == "concat":
            combined = torch.cat([h1, h2], dim=-1)
        elif self.aggregation == "abs_diff":
            combined = torch.abs(h1 - h2)
        elif self.aggregation == "division":
            combined = h1 / (h2 + 1e-8) 
        elif self.aggregation == "all":
            combined = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=-1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        logits = self.mlp(combined)
        return logits.squeeze(-1)



