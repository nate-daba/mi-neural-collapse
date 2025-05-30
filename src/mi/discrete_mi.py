import torch

def compute_mi(joint_pmf) -> float:
    """
    Compute the mutual information I(X; Y) between two discrete random variables X and Y
    given their joint probability mass function (PMF).
    
    Args:
        joint_pmf: A 2D tensor where joint_pmf[i, j] represents P(X=i, Y=j).
    
    Returns:
        float: The mutual information I(X; Y) in nats.
    """
    # Ensure we're working with a PyTorch tensor
    if not isinstance(joint_pmf, torch.Tensor):
        joint_pmf = torch.tensor(joint_pmf)
    
    # Ensure the joint PMF is a valid probability distribution
    assert torch.isclose(torch.sum(joint_pmf), torch.tensor(1.0)), "Joint PMF must sum to 1."
    
    # Compute marginal PMFs
    p_x = torch.sum(joint_pmf, dim=1, keepdim=True)  # P(X=x)
    p_y = torch.sum(joint_pmf, dim=0, keepdim=True)  # P(Y=y)
    
    # Compute the mutual information
    # Handle division by zero and log(0) cases
    ratio = joint_pmf / (p_x * p_y)
    # Replace zeros to avoid log(0)
    ratio = torch.where(joint_pmf > 0, ratio, torch.ones_like(ratio))
    mi_matrix = joint_pmf * torch.log2(ratio)
    mi_matrix = torch.where(torch.isfinite(mi_matrix), mi_matrix, torch.zeros_like(mi_matrix))
    
    return torch.sum(mi_matrix).item()