import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricContrastLoss(nn.Module):
    """
    Expression-aware triplet loss wrapper

    Supports:
        - Symmetric contrastive structure
        - Multiple triplet realizations
        - Clean dispatch (no branching in forward)

    Expected inputs
    ---------------
    G_a : Tensor [B, D]
        Generated features (expression A)

    P_a : Tensor [B, D]
        Positive GT features (expression A)

    G_b : Tensor [B, D] (optional)
        Generated features (expression B)

    P_b : Tensor [B, D] (optional)
        Positive GT features (expression B)

    Notes
    -----
    Negative samples follow your definition:
        N_b = P_b
        N_a = P_a
    """

    def __init__(
        self,
        mode: str = "hinge",
        symmetric: bool = True,
        margin: float = 0.2,
        temperature: float = 0.07,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()

        # ---------- configs ----------
        self.symmetric = symmetric
        self.margin = margin
        self.temperature = temperature
        self.eps = eps
        self.reduction = reduction

        # ---------- dispatch ----------
        triplet_map = {
            "hinge": self._triplet_hinge,
            "ratio": self._triplet_ratio,
            "nce": self._triplet_nce,
        }

        if mode not in triplet_map:
            raise ValueError(f"Unknown triplet mode: {mode}")

        self.triplet_fn = triplet_map[mode]

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(
        self,
        G_a: torch.Tensor,
        P_a: torch.Tensor,
        G_b: torch.Tensor = None,
        P_b: torch.Tensor = None,
    ):
        """
        Compute expression triplet loss
        """

        # ---------- primary branch ----------
        loss_a = self.triplet_fn(G_a, P_a, P_b)

        # ---------- symmetric branch ----------
        if self.symmetric:
            if G_b is None or P_b is None:
                raise ValueError(
                    "Symmetric loss requires G_b and P_b"
                )

            loss_b = self.triplet_fn(G_b, P_b, P_a)
            loss = 0.5 * (loss_a + loss_b)
        else:
            loss = loss_a

        # ---------- reduction ----------
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("Invalid reduction")

        return loss

    # ==========================================================
    # Triplet Realizations
    # ==========================================================

    def _triplet_hinge(self, G, P, N):
        """ 
        Hinge-based realization 
        
        Implements: 
        
            max(0, d(G,P) - d(G,N) + margin) 
            
        To be implemented later. """

        # cosine similarity
        sim_gp = F.cosine_similarity(G, P, dim=-1)
        sim_gn = F.cosine_similarity(G, N, dim=-1)

        # convert to distance
        d_gp = 1 - sim_gp
        d_gn = 1 - sim_gn

        loss = torch.relu(d_gp - d_gn + self.margin)

        return loss

    def _triplet_ratio(self, G, P, N):
        """
        Log-ratio realization

        Implements:

            log((d(G,P)+eps)/(d(G,N)+eps))

        To be implemented later.
        """

        # cosine similarity
        sim_gp = F.cosine_similarity(G, P, dim=-1)
        sim_gn = F.cosine_similarity(G, N, dim=-1)

        # convert to distance
        d_gp = 1 - sim_gp
        d_gn = 1 - sim_gn

        # ---------- log ratio
        loss = torch.log((d_gp + self.eps) / (d_gn + self.eps))

        return loss.mean()

    def _triplet_nce(self, G, P, N):
        """
        InfoNCE-style realization

        Implements:

            -log exp(sim(G,P)/tau)
                 /
                 (exp(sim(G,P)/tau)+exp(sim(G,N)/tau))

        To be implemented later.
        """
        # cosine similarity (CLIP normalized features)
        sim_gp = F.cosine_similarity(G, P, dim=-1)
        sim_gn = F.cosine_similarity(G, N, dim=-1)

        # stack logits
        logits = torch.stack([sim_gp, sim_gn], dim=1) / self.temperature

        # positive is class 0
        labels = torch.zeros(
            logits.size(0),
            dtype=torch.long,
            device=logits.device
        )

        loss = F.cross_entropy(logits, labels)

        return loss
