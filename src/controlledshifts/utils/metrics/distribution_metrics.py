import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from controlledshifts.utils.constants import EPSILON


def compute_perplexity(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the perplexity, the uncertainty of the logits with the target.

    Notation:
        B: batch size
        Q: number of queries
        V: vocabulary size

    Args:
        logits (torch.Tensor(B, Q, V)): model output logits
        target (torch.Tensor(B, Q, 1)): target value.

    Returns:
        torch.Tensor(B, 1): perplexity score.
    """
    # Convert logits to log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the correct target tokens. The gather method will pick the log probabilities of
    # the true target tokens.
    target_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    # Calculate the negative log likelihood
    negative_log_likelihood = -target_log_probs

    # Calculate the mean negative log-likelyhood
    mean_ll = negative_log_likelihood.mean()

    # Calculate perplexity as exp(mean negative log likelihood)
    return torch.exp(mean_ll)


def compute_marginal_pdf(x: torch.Tensor, y: torch.Tensor, sigma: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the marginal probability density function (PDF) given two variables.

    Notation:
        B: batch size
        Q: number of queries
        C: number of classes

    Args:
        x (torch.Tensor(B, Q, C)): a tensor representing a variable.
        y (torch.Tensor(B, Q, C)): a tensor representing a variable.
        sigma (float): standrad deviation.

    Returns:
        pdf (torch.Tensor(B, C)): probability density function of x.
        kernel_values: (torch.Tensor(B, C)): kernel density estimation of x.
    """
    # input shapes are maintained through
    residuals = x - y.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))
    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + EPSILON
    pdf = pdf / normalization
    return pdf, kernel_values


def compute_joint_pdf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the joint probability density function (PDF) between two variables.

    Args:
        x (torch.Tensor(B, C)): a tensor representing a random variable.
        y (torch.Tensor(B, C)): a tensor representing a random variable.

    Returns:
        pdf (torch.Tensor(B, C) joint probability density between the variables.
    """
    # joint kernel shape: (B, C, C)
    joint_kernel_values = torch.matmul(x.transpose(1, 2), y)
    # normalization shape: (B, 1)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + EPSILON
    # pdf shape: (B, C, C)
    return joint_kernel_values / normalization


def compute_mutual_information(x: torch.Tensor, y: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Computes the mutual information between the X and Y. I(X;Y) = H(X) + H(Y) - H(X;Y), where H(X) and H(Y) are the
    marginal entropies and H(X;Y) is the conditional entropy. Implementation based on:
    https://github.com/connorlee77/pytorch-mutual-information

    Notation:
        B: batch size
        Q: number of queries
        C: number of classes

    Args:
        x (torch.Tensor(B, C)): probability distribution over the classes.
        y (torch.Tensor(B, 1)): target value.
        normalize (bool): if True it will normalize the mutual information value.

    Returns:
        torch.Tensor(B, 1): perplexity score.
    """
    if x.shape != y.shape:
        error_message = f"Shape of x: {x.shape} != shape of y: {y.shape}"
        raise ValueError(error_message)

    num_dims = x.shape[-1]
    bins = nn.Parameter(torch.linspace(0, 1, num_dims).float(), requires_grad=False).to(x.device)

    # Compute the the marginal distribution between the probability distribution x and a uniform distribution
    pdf_x, kernel_values_x = compute_marginal_pdf(x, bins)
    # Compute the the marginal distribution between the target distribution y and a uniform distribution
    pdf_y, kernel_values_y = compute_marginal_pdf(y, bins)
    # The joint distribution between x and y
    pdf_xy = compute_joint_pdf(kernel_values_x, kernel_values_y)

    # Compute the entropies
    H_x = -torch.sum(pdf_x * torch.log2(pdf_x + EPSILON), dim=1)  # noqa: N806
    H_y = -torch.sum(pdf_y * torch.log2(pdf_y + EPSILON), dim=1)  # noqa: N806
    H_xy = -torch.sum(pdf_xy * torch.log2(pdf_xy + EPSILON), dim=(1, 2))  # noqa: N806

    # Compute the mutual information value
    mutual_information = H_x + H_y - H_xy
    if normalize:
        mutual_information = 2 * mutual_information / (H_x + H_y)
    return mutual_information
