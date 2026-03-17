import torch
from skimage import metrics

from skimage.color import rgb2gray

import cv2
class MovingAverage(object):
    def __init__(self, n):
        self.n = n
        self._cache = []
        self.mean = 0

    def update(self, val):
        self._cache.append(val)
        if len(self._cache) > self.n:
            del self._cache[0]
        self.mean = sum(self._cache) / len(self._cache)

    def get_value(self):
        return self.mean

def torch2numpy(tensor, gamma=None):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    tensor *= 255.0
    tensor=torch.clamp(tensor,0,255).to(torch.uint8)
    tensor = tensor.squeeze(1)
    return tensor.permute(0, 2, 3, 1).cpu().data.numpy()

def calculate_psnr(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    psnr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        psnr += metrics.peak_signal_noise_ratio(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...])
   
        n += 1.0
    return psnr / n

def calculate_ssim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    ssim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim += metrics.structural_similarity(rgb2gray(cv2.cvtColor(target_tf[im_idx, ...],cv2.COLOR_BGR2RGB)),
                                             rgb2gray(cv2.cvtColor(output_tf[im_idx, ...],cv2.COLOR_BGR2RGB)), data_range=1.0)
        n += 1.0
    return ssim / n
