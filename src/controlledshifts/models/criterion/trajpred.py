import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig
from torch.distributions import Laplace, MultivariateNormal

from controlledshifts.models.criterion.base_criterion import Criterion
from controlledshifts.schemas.output_schemas import ModelOutput


class TrajectoryPrediction(Criterion):
    """Trajectory prediction criterion using a GMM-based objective."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.rho_limit = config.get("rho_limit", 0.5)
        self.log_std = config.get("log_std", -1.609)
        self.log_range = config.get("log_range", 5.0)
        self.use_square_gmm = config.get("use_square_gmm", False)
        self.pre_nearest_mode_idxs = config.get("pre_nearest_mode_idxs", None)
        self.timestamp_loss_weight = config.get("timestamp_loss_weight", None)
        self.trajpred_weight = config.get("trajpred_weight", 1.0)

        self.square_gmm_size = 3
        self.nonsquare_gmm_size = 5  # (mu_x, mu_y, sigma_x, sigma_y, rho)

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute GMM-based trajectory prediction loss.

        Notation:
            B: batch size
            M: number of trajectory modes
            F: number of future steps
            Dp: prediction parameter size
                - 3: (µ_x, µ_y, log_sigma) when ``use_square_gmm`` is True
                - 5: (µ_x, µ_y, log_sigma_x, log_sigma_y, rho) otherwise
            Dg: ground-truth size (x, y, valid_flag)
            L_reg: nearest-mode regression loss
            L_cls: nearest-mode classification loss
            L_traj: final trajectory loss, ``trajpred_weight * mean(L_reg + L_cls)``

        Args:
            model_output (ModelOutput): Structured model outputs.
                - decoded_trajectories (torch.Tensor): shape (B, M, F, Dp)
                - mode_logits (torch.Tensor): shape (B, M)
                - future_ground_truth (torch.Tensor): shape (B, F, Dg)

        Returns:
            torch.Tensor: Scalar loss value ``L_traj``.
        """
        # Predicted mode logits: (B, M)
        trajectory_decoder_output = model_output.trajectory_decoder_output
        pred_scores = trajectory_decoder_output.mode_logits.value
        # Predicted trajectory parameters: (B, M, F, Dp)
        pred_trajs = trajectory_decoder_output.decoded_trajectories.value
        # Ground-truth trajectories: (B, F, Dg)
        gt_trajs = model_output.future_ground_truth.value

        if self.use_square_gmm:
            assert pred_trajs.shape[-1] == self.square_gmm_size
        else:
            assert pred_trajs.shape[-1] == self.nonsquare_gmm_size

        batch_size = pred_trajs.shape[0]
        # Ground-truth validity mask: (B, F)
        gt_valid_mask = gt_trajs[..., -1]

        if self.pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = self.pre_nearest_mode_idxs
        else:
            # Pairwise XY distances between predictions and ground truth: (B, M, F)
            distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
            # Sum distances across valid future steps: (B, M)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)
            nearest_mode_idxs = distance.argmin(dim=-1)

        # Batch indices for advanced indexing: (B)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)

        # Nearest-mode trajectory parameters: (B, F, Dp)
        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]
        # Residual XY offsets: (B, F, 2)
        res_trajs = gt_trajs[..., :2] - nearest_trajs[:, :, 0:2]
        # dx and dy residuals: (B, F)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if self.use_square_gmm:
            # All parameter tensors: (B, F)
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=self.log_std, max=self.log_range)
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=self.log_std, max=self.log_range)
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=self.log_std, max=self.log_range)
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-self.rho_limit, max=self.rho_limit)

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if self.timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * self.timestamp_loss_weight[None, :]

        # Regression loss terms: (B, F)
        # If p(x) = a^{-1} * exp(-b), then -log(p(x)) = log(a) + b.
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (B, F)
        reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * (
            (dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2)
        )  # (B, F)

        # Regression and classification losses: (B)
        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)
        loss_cls = F.cross_entropy(input=pred_scores, target=nearest_mode_idxs, reduction="none")

        # Final scalar loss.
        return self.trajpred_weight * (reg_loss + loss_cls).mean()


class TrajectoryPredictionAutoBot(Criterion):
    """Trajectory-prediction criterion used by AutoBot.

    Reference: https://arxiv.org/abs/2104.00563.
    Adapted from the UniTraj framework:
    https://github.com/vita-epfl/UniTraj/blob/main/unitraj/models/autobot/autobot.py
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the AutoBot trajectory prediction criterion."""
        super().__init__(config=config)

        # These parameters are borrowed from UniTraj and were not tuned for this project.
        # They can be tuned further for better performance.
        self.fdeade_weight = config.get("fdeade_weight", 100.0)
        self.kl_weight = config.get("kl_weight", 20.0)
        self.entropy_weight = config.get("entropy_weight", 40.0)
        self.use_fdeade_aux_loss = config.get("use_fdeade_aux_loss", True)

        self.nonsquare_gmm_size = 5  # (mu_x, mu_y, sigma_x, sigma_y, rho)

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute the trajectory prediction loss for the AutoBot model.

        Notation:
            B: batch size
            M: number of modes
            F: number of future steps
            Dp: prediction parameter size (mu_x, mu_y, sigma_x, sigma_y, rho)
            Dg: ground-truth size (x, y, valid_flag)
            L_nll: posterior-weighted NLL term
            L_ent: entropy regularization term
            L_kl: KL-divergence regularization term
            L_aux: ADE/FDE auxiliary term (optional)
            L_traj: final trajectory loss, ``L_nll + entropy_weight * L_ent + L_kl + L_aux``

        Args:
            model_output (ModelOutput): Structured model outputs.
                - decoded_trajectories (torch.Tensor): shape (B, M, F, Dp)
                - mode_probabilities (torch.Tensor): shape (B, M)
                - future_ground_truth (torch.Tensor): shape (B, F, Dg)

        Returns:
            torch.Tensor: Scalar loss value ``L_traj``.
        """
        # Extract the relevant trajectory prediction tensors.
        trajectory_decoder_output = model_output.trajectory_decoder_output
        pred_scores = trajectory_decoder_output.mode_probabilities.value  # shape: (B, M)
        pred_trajs = trajectory_decoder_output.decoded_trajectories.value  # shape: (B, M, F, Dp)
        gt_trajs = model_output.future_ground_truth.value  # shape: (B, F, Dg)
        gt_valid_mask = gt_trajs[..., -1]  # shape: (B, F)

        _, num_modes = pred_scores.shape

        log_lik_list = []
        with torch.no_grad():
            for mode in range(num_modes):
                nll = self.nll_pytorch_dist(
                    pred_trajs[:, mode], gt_trajs, gt_valid_mask, return_loss=False
                )  # shape: (B, F)
                log_lik_list.append(-nll.unsqueeze(1))  # Insert mode axis for concatenation: (B, 1, F)

            # Concatenate per-mode log-likelihoods.
            log_likelihood = torch.cat(log_lik_list, dim=1)  # shape: (B, M, F)
            log_priors = torch.log(pred_scores)  # shape: (B, M)
            log_posterior_unnorm = log_likelihood + log_priors.unsqueeze(2)  # shape: (B, M, F) + (B, M, 1) = (B, M, F)

            # Normalize over the future-step axis for numerical stability.
            log_sumexp = torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)  # shape: (B, M, 1)
            log_posterior = log_posterior_unnorm - log_sumexp  # shape: (B, M, F) - (B, M, 1) = (B, M, F)

            # Posterior-like weights used in the loss.
            post_pr = torch.exp(log_posterior)  # shape: (B, M, F)
            post_pr = post_pr.to(gt_trajs.device)

        # Compute loss.
        loss = 0.0
        for mode in range(num_modes):
            nll_k = self.nll_pytorch_dist(pred_trajs[:, mode], gt_trajs, gt_valid_mask, return_loss=True)  # shape: (B)
            loss_k = nll_k.unsqueeze(1) * post_pr[:, mode]  # shape: (B, 1) * (B, F) = (B, F)
            loss += loss_k.mean()

        # Entropy regularizer: prevent each mode from covering multiple outcomes.
        entropy_vals = [
            self.compute_bivariate_gaussian_distribution(pred_trajs[:, mode]).entropy() for mode in range(num_modes)
        ]
        entropy_vals = torch.stack(entropy_vals).permute(1, 0, 2)  # shape: (M, B, F) -> (B, M, F)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])  # shape: (B, M) -> (B) -> 1
        loss += self.entropy_weight * entropy_loss

        # KL divergence between prior mode probabilities and aggregated posterior weights.
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        kl_loss = self.kl_weight * kl_loss_fn(torch.log(pred_scores), post_pr.sum(dim=2))  # shape: (B, M) -> (B) -> 1

        # ADE/FDE auxiliary loss: L2 distance to the best-matching mode.
        if self.use_fdeade_aux_loss:
            adefde_loss = self.l2_loss_fde(pred_trajs, gt_trajs, gt_valid_mask)
        else:
            adefde_loss = torch.tensor(0.0).to(gt_trajs.device)

        return loss + kl_loss + adefde_loss

    def compute_bivariate_gaussian_distribution(self, pred: torch.Tensor) -> MultivariateNormal:
        """Build bivariate Gaussian distributions for one predicted mode.

        Notation:
            B: batch size
            F: number of future steps
            Dp = 5: prediction parameter size (mu_x, mu_y, sigma_x, sigma_y, rho)

        Args:
            pred (torch.Tensor): Predicted trajectories of shape (B, F, Dp).

        Returns:
            MultivariateNormal: Distribution with batch shape (B, F) and event shape (2).
        """
        assert pred.shape[-1] == self.nonsquare_gmm_size, (
            f"Expected last dimension of pred to be {self.nonsquare_gmm_size}, but got {pred.shape[-1]}"
        )

        mu_x = pred[:, :, 0].unsqueeze(2)  # shape: (B, F, 1)
        mu_y = pred[:, :, 1].unsqueeze(2)  # shape: (B, F, 1)
        sigma_x = pred[:, :, 2]  # shape: (B, F)
        sigma_y = pred[:, :, 3]  # shape: (B, F)
        rho = pred[:, :, 4]  # shape: (B, F)

        # Build covariance matrices for each (B, F) element.
        cov = torch.stack(
            [
                torch.stack([sigma_x**2, rho * sigma_x * sigma_y], dim=-1),  # shape: (B, F, 2)
                torch.stack([rho * sigma_x * sigma_y, sigma_y**2], dim=-1),  # shape: (B, F, 2)
            ],
            dim=-2,
        )  # shape: (B, F, 2, 2)

        # Construct a batched multivariate normal distribution.
        return MultivariateNormal(
            loc=torch.cat((mu_x, mu_y), dim=-1),
            covariance_matrix=cov,
            validate_args=False,
        )

    def compute_laplace_distribution(self, pred: torch.Tensor) -> Laplace:
        """Build factorized Laplace distributions for one predicted mode.

        Notation:
            B: batch size
            F: number of future steps
            Dp = 5: prediction parameter size (mu_x, mu_y, sigma_x, sigma_y, rho)

        Args:
            pred (torch.Tensor): Predicted trajectories of shape (B, F, Dp).

        Returns:
            Laplace: Laplace distribution with independent x/y components.
        """
        # Laplace(loc=mu, scale=sigma)
        return Laplace(loc=pred[..., :2], scale=pred[..., 2:4], validate_args=False)

    def nll_pytorch_dist(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        gt_valid_mask: torch.Tensor,
        *,
        return_loss: bool = True,
    ) -> torch.Tensor:
        """Compute the negative log-likelihood of the data under the predicted distribution.

        Args:
            pred (torch.Tensor): Predicted trajectories for one mode of shape (B, F, D=5).
            gt (torch.Tensor): Ground-truth trajectories of shape (B, F, D=3).
            gt_valid_mask (torch.Tensor): Binary mask of shape (B, F) for valid ground-truth points.
            return_loss (bool): If True, return aggregated NLL per sample with shape (B).
                If False, return per-step NLL with shape (B, F).

        Returns:
            torch.Tensor: NLL tensor with shape (B) or (B, F), depending on ``return_loss``.
        """
        laplace_dist = self.compute_laplace_distribution(pred)
        gt_xy = gt[:, :, :2]
        if return_loss:
            return ((-laplace_dist.log_prob(gt_xy)).sum(-1) * gt_valid_mask).sum(1)  # shape: (B)
        return (-laplace_dist.log_prob(gt_xy)).sum(dim=2) * gt_valid_mask  # shape: (B, F)

    def l2_loss_fde(self, pred: torch.Tensor, gt: torch.Tensor, gt_valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute the L2 loss for final displacement error (FDE) and average displacement error (ADE).

        Args:
            pred (torch.Tensor): Predicted trajectories of shape (B, M, F, D=5).
            gt (torch.Tensor): Ground-truth trajectories of shape (B, F, D=3).
            gt_valid_mask (torch.Tensor): Binary mask of shape (B, F) for valid ground-truth points.

        Returns:
            torch.Tensor: Computed L2 auxiliary loss for FDE and ADE.
        """
        diff = pred[:, :, :, :2] - gt[:, None, :, :2]  # shape: (B, M, F, 2)
        fde_loss = torch.norm(diff[:, :, -1], p=2, dim=-1) * gt_valid_mask[:, None, -1]
        ade_loss = (torch.norm(diff.transpose(1, 2), p=2, dim=-1) * gt_valid_mask.unsqueeze(-1)).mean(dim=1)
        loss, _ = (fde_loss + ade_loss).min(dim=1)
        return self.fdeade_weight * loss.mean()
