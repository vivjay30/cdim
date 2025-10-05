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
    """
    y = y.to(device=device, dtype=dtype).flatten()
    m = y.numel()

    # ----------------  tr((AA^T)^2)
    t1_acc = torch.zeros((), device=device, dtype=dtype)
    for _ in range(n_trace_samples):
        v = torch.randn(in_shape, device=device, dtype=dtype)
        w = torch.randn(in_shape, device=device, dtype=dtype)
        s = torch.dot(operator(v).flatten(), operator(w).flatten())
        t1_acc += s * s
    T1 = t1_acc / n_trace_samples

    # ----------------  y^T AA^T y
    t2_acc = torch.zeros((), device=device, dtype=dtype)
    for _ in range(n_y_samples):
        v = torch.randn(in_shape, device=device, dtype=dtype)
        s = torch.dot(y, operator(v).flatten())
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



def trace_AAt(
    operator: Callable[[torch.Tensor], torch.Tensor],
    input_shape = (1, 3, 256, 256),
    num_samples: int = 128,
    device: str = "cuda"            # or "cpu"
) -> float:
    """
    Unbiased Monte-Carlo estimate of tr(A Aᵀ) for a black-box linear operator.

    operator      : function that maps a (1,C,H,W) tensor → down-sampled tensor
    input_shape   : shape expected by the operator
    num_samples   : more samples → lower variance (error ≈ O(1/√num_samples))
    """
    total = 0.0
    for _ in range(num_samples):
        # Rademacher noise (±1).  Use torch.randn for Gaussian instead.
        z = torch.empty(input_shape, device=device).bernoulli_().mul_(2).sub_(1)
        Az = operator(z).flatten()          # output can have any shape
        total += torch.dot(Az, Az).item()   # ||Az||²
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



# new file: cdim/moments.py
import torch
from torch import Tensor
from typing import Callable

def hessian_traces(x_hat0, grad_g, operator, y, n_probe=4):
    """
    Returns   tr(H)   and   tr(H²)   for g(x)=‖A(x)-y‖²
    using Hutchinson probing.
    Requires grad_g (∇g) already computed w.r.t. x_hat0.
    """
    tr_H  = 0.0
    tr_H2 = 0.0
    for _ in range(n_probe):
        v   = torch.randn_like(x_hat0)
        Hv  = torch.autograd.grad(
                  grad_g, x_hat0, v,
                  retain_graph=True, create_graph=True)[0]
        tr_H  += (v * Hv).flatten(start_dim=1).sum(dim=1).mean()
        tr_H2 += (Hv.flatten(start_dim=1).pow(2).sum(dim=1)).mean()
    tr_H  /= n_probe
    tr_H2 /= n_probe
    return tr_H, tr_H2


def jacobian_energy(x_hat0, operator, y, n_probe=8):
    """
    Returns trace(J_r J_rᵀ) for r(x)=A(x)-y  via Hutchinson.
    """
    tr_JJ = 0.0
    r0    = operator(x_hat0) - y                # (B, …)
    for _ in range(n_probe):
        v      = torch.randn_like(r0)           # probe in *measurement* space
        # Jᵀ v  = grad_{x} ⟨r(x), v⟩
        JTv    = torch.autograd.grad(
                     r0, x_hat0, v,
                     retain_graph=True,
                     create_graph=False)[0]
        tr_JJ += JTv.flatten(start_dim=1).pow(2).sum(dim=1).mean()
    return tr_JJ / n_probe


@torch.enable_grad()
def forward_moments_nonlinear(
    x_hat0  : Tensor,
    alphabar: float,
    operator, y: Tensor,
    n_probe : int = 32
) -> tuple[float, float]:
    """
    First-order mean  +  first- & second-order variance.
    """
    # detach to cut old graph, then re-enable grad
    x_hat0 = x_hat0.detach().requires_grad_(True)

    # 0-th order residual & scalar loss
    r0 = operator(x_hat0) - y
    g0 = (r0.flatten(start_dim=1).pow(2).sum(dim=1)).mean()   # scalar

    # --------- Jacobian energy (dominant mean term) ----------
    tr_JJ = jacobian_energy(x_hat0, operator, y, n_probe=n_probe)

    mu = g0 + (1 - alphabar) * tr_JJ          # Eq. above
    mu = torch.clamp(mu, min=0.0)             # keep it ≥ 0

    # --------- variance:   first-order + ½ second-order -----------
    grad_g = torch.autograd.grad(g0, x_hat0, create_graph=True)[0]

    # Hessian traces for 2-nd order variance correction
    tr_H , tr_H2 = hessian_traces(           # same helper as previous reply
        x_hat0, grad_g, operator, y, n_probe=n_probe)

    var = ((1 - alphabar) *
           grad_g.flatten(start_dim=1).pow(2).sum(dim=1).mean()
          +0.5 * (1 - alphabar)**2 * (2 * tr_H2 + tr_H**2))
    return mu.item(), var.item()
