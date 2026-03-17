import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import models

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """VGG-based Perceptual Loss."""

    def __init__(self, layer_weights=None, vgg_type='vgg19', use_input_norm=True, range_norm=False, loss_weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if layer_weights is None:
            layer_weights = {'conv5_4': 1.0}
        self.layer_weights = layer_weights

        # VGG model
        if vgg_type == 'vgg19':
            vgg = models.vgg19(pretrained=True)
        elif vgg_type == 'vgg16':
            vgg = models.vgg16(pretrained=True)
        else:
            raise ValueError(f'Unsupported VGG type: {vgg_type}')

        # Extract features
        self.vgg_layers = vgg.features
        self.layer_name_mapping = {
            '3': "conv1_2",
            '8': "conv2_2",
            '17': "conv3_4",
            '26': "conv4_4",
            '35': "conv5_4"
        }

        # We only need layers up to the max index we use
        # Find max index
        max_idx = 0
        for name in layer_weights.keys():
            # Find index corresponding to name
            for k, v in self.layer_name_mapping.items():
                if v == name:
                    max_idx = max(max_idx, int(k))

        self.vgg_layers = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])

        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # VGG Mean/Std for normalization (ImageNet)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, gt):
        # Input x, gt should be in [0, 1]

        if self.range_norm:
            x = (x + 1) / 2
            gt = (gt + 1) / 2

        if self.use_input_norm:
            x = (x - self.mean) / self.std
            gt = (gt - self.mean) / self.std

        loss = 0.0
        x_features = x
        gt_features = gt

        for name, module in self.vgg_layers.named_children():
            x_features = module(x_features)
            gt_features = module(gt_features)

            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.layer_weights:
                    loss += F.l1_loss(x_features, gt_features) * self.layer_weights[layer_name]

        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """SSIM Loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        window_size (int): Window size for SSIM. Default: 11.
    """

    def __init__(self, loss_weight=1.0, window_size=11, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.window_size = window_size
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, img1, img2, weight=None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # SSIM Loss doesn't support weight map easily in this implementation
        # But for interface consistency we accept it
        return self.loss_weight * (1 - self._ssim(img1, img2, window, self.window_size, channel))

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    """Gradient Loss to preserve edges and textures."""

    def __init__(self, loss_weight=1.0):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        # Sobel kernel
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, target, weight=None):
        # Apply to each channel
        b, c, h, w = pred.shape
        # Flatten channels to use conv2d with groups=c
        # Or just apply to each channel separately. Simple way: reshape (B*C, 1, H, W)
        pred_flat = pred.reshape(b * c, 1, h, w)
        target_flat = target.reshape(b * c, 1, h, w)

        # Conv
        pred_gx = F.conv2d(pred_flat, self.kernel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, self.kernel_y, padding=1)

        target_gx = F.conv2d(target_flat, self.kernel_x, padding=1)
        target_gy = F.conv2d(target_flat, self.kernel_y, padding=1)

        loss_x = torch.abs(pred_gx - target_gx)
        loss_y = torch.abs(pred_gy - target_gy)
        loss = loss_x + loss_y

        if weight is not None:
            # weight shape (B, 1, H, W) -> (B*C, 1, H, W)
            # Need to match dimensions carefully
            # Assume weight is same for all channels
            weight_flat = weight.repeat(1, c, 1, 1).reshape(b * c, 1, h, w)
            loss = loss * weight_flat

        return self.loss_weight * torch.mean(loss)

@LOSS_REGISTRY.register()
class ColorLoss(nn.Module):
    """Color consistency loss using simple Gaussian blur to compare low-freq color info."""
    def __init__(self, loss_weight=1.0, patch_size=16):
        super(ColorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.patch_size = patch_size
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, pred, target):
        # Average color in patches
        pred_mean = self.pool(pred)
        target_mean = self.pool(target)
        loss = F.mse_loss(pred_mean, target_mean)
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class NPRLoss(nn.Module):
    """Color consistency loss using simple Gaussian blur to compare low-freq color info."""
    def __init__(self, loss_weight=1.0, factor=0.5,threshold=None):
        super(NPRLoss, self).__init__()
        self.loss_weight = loss_weight
        self.factor = factor
        self.threshold = threshold

    def interpolate(self, img, factor=0.5):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                             scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)

    def forward(self, pred, target):
        # Average color in patches
        target_mean = target - self.interpolate(target)
        pred_mean = pred - self.interpolate(pred)

        if self.threshold:
            mask = target_mean.abs()>self.threshold
            if mask.numel()==0:
                return 0
            loss = F.mse_loss(pred_mean[mask], target_mean[mask])
        else:
            loss = F.mse_loss(pred_mean, target_mean)

        return self.loss_weight * loss


