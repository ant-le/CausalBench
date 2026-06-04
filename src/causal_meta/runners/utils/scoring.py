import torch


class LinearGaussianScorer:
    """
    Fits a Linear Gaussian SCM to the predicted graph and observational data,
    then evaluates Negative Log Likelihood (NIL) on held-out data (observational or interventional).
    """

    def __init__(self, adjacency: torch.Tensor, obs_data: torch.Tensor):
        """
        Args:
            adjacency: (N, N) binary adjacency matrix.
            obs_data: (Batch, N) observational data used for fitting.
        """
        self.adjacency = adjacency.float()
        self.n_nodes = adjacency.shape[0]
        self.obs_data = obs_data.float()
        self.device = obs_data.device

        # Parameters (learned)
        self.weights = torch.zeros_like(adjacency)
        self.biases = torch.zeros(self.n_nodes, device=self.device)
        self.variances = torch.ones(self.n_nodes, device=self.device)

        self.fitted = False

    def fit(self):
        """Fit Linear Gaussian parameters via Least Squares."""
        num_samples = self.obs_data.shape[0]

        for j in range(self.n_nodes):
            parents = torch.nonzero(self.adjacency[:, j], as_tuple=False).flatten()

            if parents.numel() == 0:
                # No parents: Estimate mean and variance
                self.biases[j] = self.obs_data[:, j].mean()
                resid = self.obs_data[:, j] - self.biases[j]
                self.variances[j] = (resid**2).mean()
            else:
                # Parents exist: Linear Regression
                X = self.obs_data[:, parents]  # (Batch, |Pa|)
                y = self.obs_data[:, j]  # (Batch,)

                # Add bias column
                X_bias = torch.cat(
                    [X, torch.ones((num_samples, 1), device=self.device)], dim=1
                )

                # Least Squares: (X^T X)^-1 X^T y
                # Add small ridge for stability
                XtX = X_bias.T @ X_bias + 1e-6 * torch.eye(
                    X_bias.shape[1], device=self.device
                )
                Xty = X_bias.T @ y
                theta = torch.linalg.solve(XtX, Xty)

                # Store weights
                self.weights[parents, j] = theta[:-1]
                self.biases[j] = theta[-1]

                # Variance
                pred = X_bias @ theta
                resid = y - pred
                self.variances[j] = (resid**2).mean()

        # Clamp variances to avoid singularities
        self.variances.clamp_min_(1e-6)
        self.fitted = True

    def _score_nll_impl(
        self,
        data: torch.Tensor,
        intervention_target: int = -1,
        intervention_value: float = 0.0,
        *,
        normalize_by_nodes: bool = False,
    ) -> float:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        _ = intervention_value
        num_samples = data.shape[0]
        total_nll = 0.0
        scored_nodes = 0

        for j in range(self.n_nodes):
            if j == intervention_target:
                continue

            scored_nodes += 1
            parents = torch.nonzero(self.adjacency[:, j], as_tuple=False).flatten()

            if parents.numel() > 0:
                pred = data[:, parents] @ self.weights[parents, j] + self.biases[j]
            else:
                pred = self.biases[j]

            resid = data[:, j] - pred
            term1 = 0.5 * torch.log(2 * torch.pi * self.variances[j])
            term2 = (resid**2) / (2 * self.variances[j])

            nll_j = term1 + term2
            total_nll += nll_j.sum()

        if num_samples <= 0:
            return 0.0

        avg_nll = float(total_nll.item() / num_samples)
        if normalize_by_nodes and scored_nodes > 0:
            avg_nll /= float(scored_nodes)
        return avg_nll

    def score_nll(
        self,
        data: torch.Tensor,
        intervention_target: int = -1,
        intervention_value: float = 0.0,
    ) -> float:
        """
        Compute Negative Log Likelihood (NLL) of data.

        Args:
            data: (Batch, N) data to evaluate.
            intervention_target: Node index if this is interventional data. -1 for observational.
            intervention_value: Value of the intervention.

        Returns:
            float: Average NLL per sample.
        """
        return self._score_nll_impl(
            data,
            intervention_target=intervention_target,
            intervention_value=intervention_value,
            normalize_by_nodes=False,
        )

    def score_nll_per_node(
        self,
        data: torch.Tensor,
        intervention_target: int = -1,
        intervention_value: float = 0.0,
    ) -> float:
        """Compute average NLL per sample and scored node."""
        return self._score_nll_impl(
            data,
            intervention_target=intervention_target,
            intervention_value=intervention_value,
            normalize_by_nodes=True,
        )
