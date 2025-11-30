from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchvision.transforms import ToPILImage
from typing import Callable

def save_to_image(tensor, filename):
    """
    Saves a torch tensor to an image.
    The image assumed to be (1, 3, H, W)
    with values between (-1, 1)
    """
    to_save = (tensor[0] + 1) / 2
    to_save = to_save.clamp(0, 1)

    # Convert to PIL Image
    transform = ToPILImage()
    img = transform(to_save)

    # Save the image
    img.save(filename)


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


@torch.no_grad()
def estimate_variance(
    operator: Callable[[Tensor], Tensor],
    y: Tensor,                    # Ax_0 + noise  (shape (m,))
    alphabar_t: float,
    in_shape: tuple[int, ...],    # e.g. (1, 3, 256, 256)
    trace: float,                 # tr(AA^T)
    sigma_y: float,
    n_trace_samples: int = 64,
    n_y_samples: int = 64,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> float:
    """
    Monte-Carlo estimator of  Var(||A x_t – y||^2)  without access to A^T.
    
    Note: For inpainting operators with a 'select' method, this computes variance
    over only the observed pixels.
    """
    use_select = hasattr(operator, 'select')
    
    # For inpainting, select only observed pixels from y
    if use_select:
        y_selected = operator.select(y).flatten()
        y = y_selected.to(device=device, dtype=dtype)
    else:
        y = y.to(device=device, dtype=dtype).flatten()
    m = y.numel()

    # ----------------  tr((AA^T)^2)
    t1_acc = torch.zeros((), device=device, dtype=dtype)
    for _ in range(n_trace_samples):
        v = torch.randn(in_shape, device=device, dtype=dtype)
        w = torch.randn(in_shape, device=device, dtype=dtype)
        if use_select:
            Av = operator.select(v).flatten()
            Aw = operator.select(w).flatten()
        else:
            Av = operator(v).flatten()
            Aw = operator(w).flatten()
        s = torch.dot(Av, Aw)
        t1_acc += s * s
    T1 = t1_acc / n_trace_samples

    # ----------------  y^T AA^T y
    t2_acc = torch.zeros((), device=device, dtype=dtype)
    for _ in range(n_y_samples):
        v = torch.randn(in_shape, device=device, dtype=dtype)
        if use_select:
            Av = operator.select(v).flatten()
        else:
            Av = operator(v).flatten()
        s = torch.dot(y, Av)
        t2_acc += s * s
    T2 = t2_acc / n_y_samples

    # ----------------  assemble variance
    alpha_bar = torch.as_tensor(alphabar_t, dtype=dtype, device=device)
    sigma2     = torch.as_tensor(sigma_y**2,  dtype=dtype, device=device)

    a2 = (torch.sqrt(alpha_bar) - 1.0).pow(2)  # (1-√ᾱ)^2
    b  = 1.0 - alpha_bar                       # (1-ᾱ)

    var = 2 * (b*b * T1 + 2 * b * sigma2 * trace + m * sigma2.pow(2)) \
        + 4 * a2 * (b * (T2 - sigma2 * trace) + sigma2 * (y.pow(2).sum() - m * sigma2))
    return var.item()



def compute_operator_distance(
    operator: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    squared: bool = True
) -> Tensor:
    """
    Compute ||Ax - y||^2 (or ||Ax - y|| if squared=False).
    
    For inpainting operators with a 'select' method, this computes the distance
    over only the observed pixels. Otherwise uses the standard operator call.
    
    Args:
        operator: The forward operator A
        x: Input tensor (e.g., image)
        y: Measurement tensor (for inpainting, this should be the full masked measurement)
        squared: If True, returns squared L2 norm. If False, returns L2 norm.
    
    Returns:
        Scalar tensor representing the distance
    """
    if hasattr(operator, 'select'):
        # Use select method for inpainting operators
        # Both x and y need to be selected to extract only observed pixels
        Ax = operator.select(x).flatten()
        y_selected = operator.select(y).flatten()
    else:
        # Standard operator application
        Ax = operator(x).flatten()
        y_selected = y.flatten()
    
    diff = Ax - y_selected
    if squared:
        return (diff ** 2).sum()
    else:
        return torch.sqrt((diff ** 2).sum())


def compute_pearson_energy(
    operator: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    noise_function,
    eps: float = 1.0,
) -> Tensor:
    """
    Compute Pearson residual energy for Poisson noise:
        R_t = sum_i (Ax[i] - y[i])^2 / Var(y[i])
    
    For Poisson noise, Var(y) is proportional to the expected count.
    This "whitens" the heteroscedastic Poisson noise.
    
    Args:
        operator: The forward operator A
        x: Input tensor (e.g., image x_t)
        y: Measurement tensor (noisy observation)
        noise_function: The Poisson noise function with get_weights method
        eps: Small constant to avoid division by zero for dark pixels
    
    Returns:
        Scalar tensor representing the Pearson residual energy
    """
    if hasattr(operator, 'select'):
        Ax = operator.select(x).flatten()
        y_selected = operator.select(y).flatten()
    else:
        Ax = operator(x).flatten()
        y_selected = y.flatten()
    
    # Get weights W = 1 / Var(y)
    weights = noise_function.get_weights(y_selected.view_as(y_selected), eps=eps)
    weights = weights.flatten()
    
    diff = Ax - y_selected
    # Pearson residual: sum of (diff^2 * weight)
    return (diff ** 2 * weights).sum()


def trace_weighted_AAt(
    operator: Callable[[torch.Tensor], torch.Tensor],
    y: Tensor,
    noise_function,
    input_shape=(1, 3, 256, 256),
    num_samples: int = 256,
    device: str = "cuda",
    eps: float = 1.0,
) -> float:
    """
    Estimate tr(A^T W A) where W = diag(1/Var(y)) for Poisson noise.
    
    Uses Hutchinson's trace estimator:
        tr(A^T W A) = E[z^T A^T W A z] where z ~ N(0, I)
    
    This is needed for the mean formula in Poisson CDIM.
    """
    use_select = hasattr(operator, 'select')
    
    # Get the weights for y
    if use_select:
        y_selected = operator.select(y).flatten()
    else:
        y_selected = y.flatten()
    
    weights = noise_function.get_weights(y_selected, eps=eps)
    weights = weights.flatten().to(device=device)
    
    total = 0.0
    for _ in range(num_samples):
        z = torch.randn(input_shape, device=device)
        if use_select:
            Az = operator.select(z).flatten()
        else:
            Az = operator(z).flatten()
        
        # Compute z^T A^T W A z = (W^{1/2} A z)^T (W^{1/2} A z) = ||W^{1/2} A z||^2
        weighted_Az = Az * torch.sqrt(weights)
        total += torch.dot(weighted_Az, weighted_Az).item()
    
    return total / num_samples


def compute_weighted_y_squared(
    y: Tensor,
    noise_function,
    operator=None,
    eps: float = 1.0,
) -> float:
    """
    Compute Σ y² * W where W = 1/Var(y) for Poisson noise.
    
    This is the correct bias energy term for Pearson residuals:
        Σ y[i]² * (255*rate)² / counts[i]
    
    NOT the same as Σ counts!
    """
    use_select = operator is not None and hasattr(operator, 'select')
    
    if use_select:
        y_selected = operator.select(y).flatten()
    else:
        y_selected = y.flatten()
    
    # Get weights W = (255*rate)² / counts
    weights = noise_function.get_weights(y_selected, eps=eps)
    
    # Compute y² * W and sum
    y_sq_weighted = y_selected ** 2 * weights
    return y_sq_weighted.sum().item()


@torch.no_grad()
def estimate_poisson_variance(
    operator: Callable[[Tensor], Tensor],
    y: Tensor,
    noise_function,
    alphabar_t: float,
    in_shape: tuple[int, ...],
    trace_AtWA: float,
    device: torch.device | str = "cuda",
    eps: float = 1.0,
) -> float:
    """
    Estimate Var(R_t) for Pearson residual energy under Poisson noise.
    
    Uses the corrected formula that includes the non-centrality term:
    
        σ_t² ≈ 2N + 4(1-α̅_t) Tr(A^T W A) + 4(a-1)² Σ(y² * W)
    
    Where:
        - N = number of observed pixels
        - a = √α̅_t
        - W = diag(1/Var(y)) = diag((255*rate)²/counts)
        - Σ(y² * W) = weighted squared observation (the non-centrality parameter)
    
    The three terms represent:
        1. Base variance of chi-squared with N degrees of freedom
        2. Diffusion noise contribution
        3. Non-centrality/bias variance (critical for high t)
    """
    use_select = hasattr(operator, 'select')
    
    if use_select:
        y_selected = operator.select(y).flatten()
    else:
        y_selected = y.flatten()
    
    N = y_selected.numel()  # Number of observed pixels
    
    a = alphabar_t ** 0.5
    b_sq = 1.0 - alphabar_t  # (1 - α̅_t)
    
    # Compute the weighted squared sum: Σ y² * W (non-centrality parameter)
    y_sq_weighted = compute_weighted_y_squared(y, noise_function, operator, eps)
    
    # Three variance terms:
    # 1. Base chi-squared variance: 2N
    term1 = 2.0 * N
    
    # 2. Diffusion contribution: 4(1-α̅_t) Tr(A^T W A)
    term2 = 4.0 * b_sq * trace_AtWA
    
    # 3. Non-centrality term: 4(a-1)² Σ(y² * W)
    term3 = 4.0 * (a - 1.0) ** 2 * y_sq_weighted
    
    variance = term1 + term2 + term3
    return variance


def compute_poisson_target_distance(
    y: Tensor,
    noise_function,
    operator,
    alphabar_t: float,
    trace_AtWA: float,
    eps: float = 1.0,
) -> float:
    """
    Compute the expected Pearson residual energy μ_t(y) for Poisson noise.
    
    μ_t = (a-1)² Σ(y² * W) + (1-α̅_t) Tr(A^T W A) + N
    
    Where:
        - a = √α̅_t
        - W = diag((255*rate)²/counts) = 1/Var(y)
        - Σ(y² * W) = weighted squared observation (bias energy in whitened space)
        - N = number of observed pixels
        - Tr(A^T W A) = weighted trace
    
    The three terms represent:
        1. Bias energy: signal attenuation from diffusion (in whitened space)
        2. Diffusion noise energy: noise from the diffusion process
        3. Measurement noise energy: baseline from Poisson measurement noise
    
    Note: The bias energy is Σ y² * W, NOT Σ counts! This is because in whitened
    space, the squared bias is (Ax_0)² / Var(y) ≈ y² * W.
    """
    use_select = hasattr(operator, 'select')
    
    if use_select:
        y_selected = operator.select(y).flatten()
    else:
        y_selected = y.flatten()
    
    N = y_selected.numel()
    
    a = alphabar_t ** 0.5
    b_sq = 1.0 - alphabar_t
    
    # Compute weighted squared sum: Σ y² * W (correct bias energy term)
    y_sq_weighted = compute_weighted_y_squared(y, noise_function, operator, eps)
    
    # Three mean terms:
    # 1. Bias energy: (a-1)² Σ(y² * W)
    term1 = (a - 1.0) ** 2 * y_sq_weighted
    
    # 2. Diffusion noise energy: (1-α̅_t) Tr(A^T W A)
    term2 = b_sq * trace_AtWA
    
    # 3. Measurement noise energy: N (since Pearson residuals normalize by variance)
    term3 = float(N)
    
    return term1 + term2 + term3


def trace_AAt(
    operator: Callable[[torch.Tensor], torch.Tensor],
    input_shape = (1, 3, 256, 256),
    num_samples: int = 256,
    device: str = "cuda"            # or "cpu"
) -> float:
    """
    Unbiased Monte-Carlo estimate of tr(A Aᵀ) for a black-box linear operator.

    operator      : function that maps a (1,C,H,W) tensor → down-sampled tensor
    input_shape   : shape expected by the operator
    num_samples   : more samples → lower variance (error ≈ O(1/√num_samples))
    
    Note: For inpainting operators with a 'select' method, this computes the trace
    over only the observed pixels, not the full tensor with zeros.
    """
    total = 0.0
    use_select = hasattr(operator, 'select')
    
    for _ in range(num_samples):
        # Rademacher noise (±1).  Use torch.randn for Gaussian instead.
        z = torch.empty(input_shape, device=device).bernoulli_().mul_(2).sub_(1)
        if use_select:
            Az = operator.select(z).flatten()   # only observed pixels
        else:
            Az = operator(z).flatten()          # output can have any shape
        total += torch.dot(Az, Az).item()       # ||Az||²
    return total / num_samples


# def trace_AAt_squared(
#     operator: Callable[[torch.Tensor], torch.Tensor],
#     input_shape: tuple = (1, 3, 256, 256),
#     num_samples: int = 32,
#     device: str = "cuda") -> float:
#     """
#     Estimates tr((A Aᵀ)^2) using Hutchinson's method and autograd for Aᵀ.
#     """
#     total = 0.0
#     for _ in range(num_samples):
#         # Sample z ~ N(0, I) (same shape as operator's *output*)
#         z = torch.randn(operator(torch.zeros(input_shape, device=device)).shape, device=device)
        
#         # Compute Aᵀz via gradient: ∇_w [⟨operator(w), z⟩] = Aᵀz
#         w = torch.randn(input_shape, device=device, requires_grad=True)
#         Az = operator(w).flatten()
#         loss = torch.dot(Az, z.flatten())  # ⟨Az, z⟩ = ⟨w, Aᵀz⟩
#         A_adj_z = torch.autograd.grad(loss, w, retain_graph=False)[0]
        
#         # Compute AAᵀz = operator(Aᵀz)
#         AA_adj_z = operator(A_adj_z.detach()).flatten()
#         total += torch.dot(AA_adj_z, AA_adj_z).item()  # ||AAᵀz||²
#     return total / num_samples


# def compute_yAAy(
#     operator: Callable[[torch.Tensor], torch.Tensor],
#     y: torch.Tensor,
#     input_shape: tuple = (1, 3, 256, 256),
#     device: str = "cuda") -> float:
#     """
#     Computes yᵀ (A Aᵀ) y using autograd to get Aᵀy.
#     """
#     # Compute Aᵀy via gradient: ∇_w [⟨operator(w), y⟩] = Aᵀy
#     w = torch.randn(input_shape, device=device, requires_grad=True)
#     Az = operator(w).flatten()
#     loss = torch.dot(Az, y.flatten())
#     A_adj_y = torch.autograd.grad(loss, w, retain_graph=False)[0]
    
#     # Compute A Aᵀ y = operator(Aᵀy)
#     AA_adj_y = operator(A_adj_y.detach()).flatten()
#     return torch.dot(AA_adj_y, y.flatten()).item()

# def variance_Axt_minus_y_sq(
#     operator: Callable[[torch.Tensor], torch.Tensor],
#     y: torch.Tensor,
#     alphabar_t: float,
#     input_shape: tuple = (1, 3, 256, 256),
#     num_samples_trace: int = 32,
#     device: str = "cuda"
# ) -> float:
#     """
#     Computes Var(||A𝐱ₜ - y||²) = 2(1-ᾱₜ)² tr((AAᵀ)²) + 4(1-ᾱₜ)(√ᾱₜ -1)² yᵀAAᵀy.
#     """
#     # Term 1: 2(1-ᾱₜ)^2 * tr((AAᵀ)^2)
#     tr_AAt_sq = trace_AAt_squared(operator, input_shape, num_samples_trace, device)
#     term1 = 2 * (1 - alphabar_t)**2 * tr_AAt_sq
    
#     # Term 2: 4(1-ᾱₜ)(√ᾱₜ -1)^2 * yᵀAAᵀy
#     yAAy = compute_yAAy(operator, y, input_shape, device)
#     term2 = 4 * (1 - alphabar_t) * (torch.sqrt(torch.tensor(alphabar_t)) - 1)**2 * yAAy
    
#     return term1 + term2




