import torch
import random
import math


def corrupt(x, amount, sde='ddpm', max_sigma=5):
    """
    Corrupt the input `x` by mixing it with noise according to `amount`.
    If `sde` is 'ddpm', use the diffusion-to-self corruption process.
    If `sde` is 've', use the variance exploding SDE process.

    Parameters:
    x (torch.Tensor): The input tensor to be corrupted.
    amount (torch.Tensor): The amount of corruption.
    sde (str, optional): The type of corruption process ('ddpm' or 've'). Defaults to 'ddpm'.
    max_sigma (float, optional): The maximum sigma for the 've' SDE process. Defaults to 1348.

    Returns:
    torch.Tensor: The corrupted tensor.
    """
    noise = torch.randn_like(x)  # Gaussian noise
    amount = amount.view(-1, 1, 1, 1)  # Reshape for broadcasting

    if sde == 'ddpm':
        # Corruption process with square roots (diffusion-to-self)
        return torch.sqrt(1 - amount) * x + torch.sqrt(amount) * noise
    elif sde == 've':
        # Variance exploding SDE process
        scaling_factor = amount * max_sigma
        return x + scaling_factor * noise
    else:
        raise ValueError(f"Unknown sde type: {sde}")


def create_block_mask_2d(batch_size, height, width, mask_percentage, block_size=16):
    """
    Create a 2D block mask for MDAE training.
    Returns mask of shape [B, 1, H, W] where 1=visible, 0=masked.
    """
    h_blocks = height // block_size
    w_blocks = width // block_size

    # Number of blocks to KEEP visible
    total_blocks = h_blocks * w_blocks
    num_visible = max(1, int(total_blocks * (1 - mask_percentage)))

    mask = torch.zeros(batch_size, 1, h_blocks, w_blocks)
    for b in range(batch_size):
        indices = torch.randperm(total_blocks)[:num_visible]
        mask[b, 0].view(-1)[indices] = 1.0

    # Upsample to full resolution using nearest neighbor
    mask = mask.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)

    # Handle case where H/W not perfectly divisible by block_size
    mask = mask[:, :, :height, :width]

    return mask  # [B, 1, H, W], 1=visible, 0=masked


# def corrupt(x, amount, diffusion2self=True):
#     """
#     Corrupt the input `x` by mixing it with noise according to `amount`.
#     If `diffusion2self` is None, randomly choose the corruption process.
    
#     Parameters:
#     x (torch.Tensor): The input tensor to be corrupted.
#     amount (torch.Tensor): The amount of corruption.
#     diffusion2self (bool, optional): Whether to use the new corruption process.
#         If None, the method is chosen randomly (default None).
    
#     Returns:
#     torch.Tensor: The corrupted tensor.
#     """
#     noise = torch.randn_like(x)  # Gaussian noise
#     amount = amount.view(-1, 1, 1, 1)  # Reshape for broadcasting

#     if diffusion2self is None:
#         diffusion2self = random.choice([True, False])

#     if diffusion2self:
#         # Corruption process with square roots
#         return torch.sqrt(1 - amount) * x + torch.sqrt(amount) * noise
#     else:
#         # Original corruption process
#         return x * (1 - amount) + noise * amount

# def corrupt(x, amount, diffusion2self=True):
#     """
#     Corrupt the input `x` by mixing it with noise according to `amount`.
#     If `diffusion2self` is True, use the corruption process with square roots.
    
#     Parameters:
#     x (torch.Tensor): The input tensor to be corrupted.
#     amount (torch.Tensor): The amount of corruption.
#     diffusion2self (bool): Whether to use the new corruption process (default False).
    
#     Returns:
#     torch.Tensor: The corrupted tensor.
#     """
#     noise = torch.randn_like(x)  # Gaussian noise
#     amount = amount.view(-1, 1, 1, 1)  # Reshape for broadcasting

#     if diffusion2self:
#         # corruption process with square roots
#         # return torch.sqrt(amount) * x + torch.sqrt(1 - amount) * noise
#         return torch.sqrt(1-amount) * x + torch.sqrt(amount) * noise # ammout = 0 preserves x

#     else:
#         # Original corruption process
#         return x * (1 - amount) + noise * amount

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)