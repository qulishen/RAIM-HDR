import os
import csv
import xml.etree.ElementTree as etree
from typing import Sequence

# from imageio import imread
from PIL import Image, ImageOps
import numpy as np
import glob
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import sys
sys.path.append('..')
from .masking_utils import *
from .cv_utils import *
import h5py
import numpy as np
from matplotlib import pyplot as plt
import fastmri
from fastmri.data import transforms as T
import os
import glob
from pathlib import Path
from tqdm import tqdm, trange

def read_h5(file_name):
    hf = h5py.File(file_name)
    volume_kspace = hf['kspace'][()]
    slice_kspace = volume_kspace
    slice_kspace2 = T.to_tensor(slice_kspace)
    slice_image = fastmri.ifft2c(slice_kspace2)
    slice_image_abs = fastmri.complex_abs(slice_image)
    slice_image_rss = fastmri.rss(slice_image_abs, dim=1)
    slice_image_rss = np.abs(slice_image_rss.numpy())
    slice_image_rss = normal(slice_image_rss)
    return slice_image_rss
    
def normal(x):
    """Normalize data to [0, 1] range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

 


def normalize_tensor(x):
    """
    Normalize a tensor to [0,1] range for better visualization and comparison.
    
    Args:
        x: Numpy array or tensor to normalize
        
    Returns:
        Normalized array/tensor with same shape as input
    """
    # Get min and max values
    min_val = np.min(x)
    max_val = np.max(x)
    
    # Handle constant-valued tensors
    if max_val == min_val:
        return x - min_val  # Return zeros if constant value
    
    # Otherwise normalize to [0,1] range
    return (x - min_val) / (max_val - min_val + 1e-8)

def get_params_3d(img_shape, output_size, n):
    d, h, w = img_shape
    td, th, tw = output_size

    # Generate random starting points
    i_list = [random.randint(0, h - th) for _ in range(n)]
    j_list = [random.randint(0, w - tw) for _ in range(n)]
    k_list = [random.randint(0, d - td) for _ in range(n)]

    return k_list, i_list, j_list, td, th, tw

def n_random_crops_3d(img, k, i, j, d, h, w):
    crops = []
    for idx in range(len(k)):
        # Define the coordinates for the crop
        start_d, start_h, start_w = k[idx], i[idx], j[idx]
        end_d, end_h, end_w = start_d + d, start_h + h, start_w + w

        # Crop and append
        new_crop = img[:, start_d:end_d, start_h:end_h, start_w:end_w]
        crops.append(new_crop)

    return crops

def n_random_crops(img, x, y, h, w):
    crops = []
    for i in range(len(x)):
        new_crop = img[y[i]:y[i] + h, x[i]:x[i] + w]
        crops.append(new_crop)
    return tuple(crops)


##############################################


def read_h5(path):
    """Read h5 file data."""
    with h5py.File(path, 'r') as f:
        data = f['data'][:]
    return data


def normalize_tensor(x):
    """
    Normalize a tensor to [0,1] range for better visualization and comparison.
    
    Args:
        x: Numpy array or tensor to normalize
        
    Returns:
        Normalized array/tensor with same shape as input
    """
    # Get min and max values
    min_val = np.min(x)
    max_val = np.max(x)
    
    # Handle constant-valued tensors
    if max_val == min_val:
        return x - min_val  # Return zeros if constant value
    
    # Otherwise normalize to [0,1] range
    return (x - min_val) / (max_val - min_val + 1e-8)


class TrainSet(Dataset):
    """
    Training dataset for color images with effective augmentation.
    """
    def __init__(self, args):
        super(TrainSet, self).__init__()
        
        # Parse parameters from args
        self.sigma = args.noise_sig
        self.patch_size = args.patch_size if hasattr(args, 'patch_size') else 128
        self.phase = 'train'  # Default to train mode
        if hasattr(args, 'phase'):
            self.phase = args.phase
        self.args = args
        
        # Find all image files in the directory
        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
        self.img_list = []
        for ext in img_extensions:
            self.img_list.extend(glob.glob(os.path.join(args.traindata_root, ext)))
        self.img_list = sorted(self.img_list)[:]
        
        print(f'Found {len(self.img_list)} images for {self.phase}')
        
        # Store original images instead of patches
        self.images = []
        
        # Load all images into memory
        for img_path in tqdm(self.img_list, desc=f'Loading {self.phase} images'):
            try:
                # Read the image with color information
                orig_img = cv2.imread(img_path, 1)
                if orig_img is None:
                    print(f"Warning: Could not read image {img_path}, skipping.")
                    continue
                    
                # Convert from BGR to RGB (OpenCV loads as BGR)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                
                # Resize image if it's too small for patching
                s = orig_img.shape
                if s[0] <= self.patch_size or s[1] <= self.patch_size:
                    orig_img = cv2.resize(orig_img, (self.patch_size*2, self.patch_size*2))
                
                # Store the original image
                self.images.append(orig_img)
            
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        print(f'Training data loading complete. Total images: {len(self.images)}')
        
    
    def __len__(self):
        """
        Return the total number of images in the dataset, with repetition.
 
        """
        if self.phase == 'train':
            return len(self.images)
        else:
            return len(self.images)
    
    def _get_index(self, idx):
        """Map from expanded index space to actual image index"""
        if self.phase == 'train':
            return idx % len(self.images)
        else:
            return idx
    
    def get_patch(self, *args, patch_size=96, scale=1, multi=False, input_large=False):
        """Extract patches from images with the same logic as reference code"""
        ih, iw = args[0].shape[:2]

        # Calculate patch sizes
        if not input_large:
            p = scale if multi else 1
            tp = p * patch_size
            ip = tp // scale
        else:
            tp = patch_size
            ip = patch_size

        # Random patch location
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        if not input_large:
            tx, ty = scale * ix, scale * iy
        else:
            tx, ty = ix, iy

        # Extract patches
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]

        return ret
    
    def augment(self, *args, hflip=True, rot=True):
        """Apply augmentations to images with the same logic as reference code"""
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        
        def _augment(img):
            if hflip: img = img[:, ::-1, :]  # Horizontal flip
            if vflip: img = img[::-1, :, :]  # Vertical flip
            if rot90: img = img.transpose(1, 0, 2)  # 90-degree rotation
            return img

        return [_augment(a) for a in args]
    
    def np2Tensor(self, *args):
        """Convert NumPy arrays to PyTorch tensors with the same logic as reference code"""
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            return tensor

        return [_np2Tensor(a) for a in args]
    
    def __getitem__(self, idx):
        """Get a sample pair with proper noise and augmentation"""
        # Get actual image index with repetition pattern
        index = self._get_index(idx)
        
        # Get clean image
        clean_img = self.images[index].copy()  # Make a copy to avoid modifying original
        
        # Create noisy version with clearly visible noise
        # Using self.sigma as a proportion of the data range (typically 255 for images)
        # Convert to float32 for proper noise addition
        clean_float = clean_img.astype(np.float32)
        noise = np.random.normal(0, self.sigma, clean_img.shape).astype(np.float32)
        noisy_img = np.clip(clean_float + noise, 0, 255).astype(np.uint8)
        
        # Apply patch extraction and augmentation (only in train or valid phase)
        if self.phase == 'train' or self.phase == 'valid':
            # Extract patches
            clean_patch, noisy_patch = self.get_patch(
                clean_img, noisy_img,
                patch_size=self.patch_size,
                scale=1,
                multi=False,
                input_large=False
            )
            
            # Apply augmentation - SAME patch gets same transformations
            clean_patch, noisy_patch = self.augment(clean_patch, noisy_patch)
        else:
            clean_patch, noisy_patch = clean_img, noisy_img
        
        # Convert to tensors
        clean_tensor = self.np2Tensor(clean_patch)[0]
        noisy_tensor = self.np2Tensor(noisy_patch)[0]
        
        # Scale to [0,1] range
        clean_tensor = clean_tensor / 255.0
        noisy_tensor = noisy_tensor / 255.0
        
        # Return simplified dictionary without paths
        return {
            'images': noisy_tensor,  # Noisy input (with clearly added noise)
            'plabels': clean_tensor   # Clean target
        }

class TestSet(Dataset):
    """
    Test/validation dataset for Noise2Score model that preserves color channels.
    """
    def __init__(self, args):
        super(TestSet, self).__init__()
        
        # Parse parameters from args
        self.sigma = args.noise_sig if hasattr(args, 'noise_sig') else 25
        
        # Fix for target_size parameter
        if hasattr(args, 'target_size'):
            if isinstance(args.target_size, tuple) and len(args.target_size) == 2:
                self.target_size = args.target_size
            elif isinstance(args.target_size, int):
                # If it's a single integer, make it a square
                self.target_size = (args.target_size, args.target_size)
            else:
                # Default to 256x256
                print(f"Warning: Invalid target_size format. Using default (256, 256).")
                self.target_size = (256, 256)
        else:
            self.target_size = (256, 256)
            
        print(f"Using target size (center crop): {self.target_size}")

        # Whether to convert to grayscale or keep RGB
        self.use_grayscale = args.use_grayscale if hasattr(args, 'use_grayscale') else False
        
        # Handle different data modalities
        if hasattr(args, 'modal') and hasattr(args, 'testdata_root'):
            # Use data path based on modality
            # Get both png and jpg files
            png_files = glob.glob(os.path.join(args.testdata_root, '*.png'))
            jpg_files = glob.glob(os.path.join(args.testdata_root, '*.jpg'))
            jpeg_files = glob.glob(os.path.join(args.testdata_root, '*.jpeg'))
            self.img_list = sorted(png_files + jpg_files + jpeg_files)
            print(f"Loading {args.modal} modality from {args.testdata_root}")
        elif hasattr(args, 'testdata_root'):
            # No specific modality mentioned
            # Get both png and jpg files
            png_files = glob.glob(os.path.join(args.testdata_root, '*.png'))
            jpg_files = glob.glob(os.path.join(args.testdata_root, '*.jpg'))
            jpeg_files = glob.glob(os.path.join(args.testdata_root, '*.jpeg'))
            self.img_list = sorted(png_files + jpg_files + jpeg_files)
            print(f"Loading images from {args.testdata_root}")
        else:
            raise ValueError("testdata_root must be specified in args")
        
        # Optional pre-computed noisy images path
        self.noisy_img_list = None
        if hasattr(args, 'eval_noised_data_path') and args.eval_noised_data_path:
            self.noisy_img_list = sorted(glob.glob(os.path.join(args.eval_noised_data_path, '*.png')))
        
        print(f'Found {len(self.img_list)} test images')
        print(f'Color mode: {"Grayscale" if self.use_grayscale else "Color (RGB)"}')
        
        # Storage for processed images
        self.clean_images = []
        self.noisy_images = []
        
        # Process each image
        for i, img_path in enumerate(tqdm(self.img_list, desc='Processing test images')):
            try:
                # Load clean image
                orig_img = cv2.imread(img_path, 1)
                if orig_img is None:
                    print(f"Warning: Could not read image {img_path}, skipping.")
                    continue
                
                # Center crop to target size (resize up first if too small)
                width, height = self.target_size
                h, w = orig_img.shape[:2]
                if h < height or w < width:
                    scale = max(height / h, width / w)
                    orig_img = cv2.resize(orig_img, (int(w * scale) + 1, int(h * scale) + 1))
                    h, w = orig_img.shape[:2]
                top = (h - height) // 2
                left = (w - width) // 2
                orig_img = orig_img[top:top+height, left:left+width, :]

                # Convert from BGR to RGB (OpenCV loads as BGR)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

                # Get or create noisy version
                if self.noisy_img_list and i < len(self.noisy_img_list):
                    # Use pre-computed noisy image
                    noisy_img = cv2.imread(self.noisy_img_list[i], 1)
                    if noisy_img is None:
                        # Fall back to generating noise
                        noise = np.random.normal(0, self.sigma, orig_img.shape)
                        noisy_img = np.clip(orig_img + noise, 0, 255).astype(np.uint8)
                    else:
                        # Center crop noisy image to match
                        nh, nw = noisy_img.shape[:2]
                        if nh < height or nw < width:
                            scale = max(height / nh, width / nw)
                            noisy_img = cv2.resize(noisy_img, (int(nw * scale) + 1, int(nh * scale) + 1))
                            nh, nw = noisy_img.shape[:2]
                        ntop = (nh - height) // 2
                        nleft = (nw - width) // 2
                        noisy_img = noisy_img[ntop:ntop+height, nleft:nleft+width, :]
                        # Convert to RGB
                        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
                else:
                    # Generate noisy version
                    noise = np.random.normal(0, self.sigma, orig_img.shape)
                    noisy_img = np.clip(orig_img + noise, 0, 255).astype(np.uint8)
                
                # Convert to PyTorch format (channels first)
                orig_img = np.transpose(orig_img, (2, 0, 1)).astype(np.float32) / 255.0
                noisy_img = np.transpose(noisy_img, (2, 0, 1)).astype(np.float32) / 255.0
                
                if self.use_grayscale:
                    # Convert to grayscale by taking mean across channels
                    orig_img = np.mean(orig_img, axis=0, keepdims=True)
                    noisy_img = np.mean(noisy_img, axis=0, keepdims=True)
                
                # Store processed images
                self.clean_images.append(orig_img)
                self.noisy_images.append(noisy_img)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                # Print more details for debugging
                if 'orig_img' in locals():
                    print(f"Image shape: {orig_img.shape if 'orig_img' in locals() else 'unknown'}")
                print(f"Target size: {self.target_size}, Type: {type(self.target_size)}")
                continue
        
        if len(self.clean_images) == 0:
            raise ValueError("No valid images found in the dataset. Please check image paths and formats.")
            
        print(f'Test dataset preparation complete. Total images: {len(self.clean_images)}')
        
        # Print shapes to verify
        if len(self.clean_images) > 0:
            print(f"Sample clean image shape: {self.clean_images[0].shape}")
            print(f"Sample noisy image shape: {self.noisy_images[0].shape}")
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        """
        Returns a sample dictionary with the expected tensor format.
        """
        # Get clean and noisy images for the specified index
        clean_img = self.clean_images[idx]
        noisy_img = self.noisy_images[idx]
        
        # Create the return dictionary
        sample = {
            'images': noisy_img,  # [C, H, W] format (C=3 for RGB, C=1 for grayscale)
            'labels': normalize_tensor(clean_img)  # [C, H, W] format, normalized to [0,1]
        }
        
        # Convert numpy arrays to torch tensors if they aren't already
        for key in sample:
            if not isinstance(sample[key], torch.Tensor):
                sample[key] = torch.from_numpy(sample[key]).float()
        
        return sample
    



class TestSet_Challenge(Dataset):
    """
    Memory-efficient test/validation dataset that loads images dynamically.
    Supports nested folder structures.
    Allows maintaining original image resolution when enable_full_image is True.
    Includes filename in the output when in challenge mode.
    In challenge mode, doesn't add noise as input images are already noisy.
    """
    def __init__(self, args):
        super(TestSet_Challenge, self).__init__()
        
        # Parse parameters from args
        self.sigma = args.noise_sig if hasattr(args, 'noise_sig') else 50
        self.enable_full_image = args.enable_full_image if hasattr(args, 'enable_full_image') else False
        
        # Add challenge mode flag
        self.challenge_mode = args.challenge if hasattr(args, 'challenge') else False
        print(f"Challenge mode: {self.challenge_mode}!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Fix for target_size parameter (used when enable_full_image is False)
        if hasattr(args, 'target_size'):
            if isinstance(args.target_size, tuple) and len(args.target_size) == 2:
                self.target_size = args.target_size
            elif isinstance(args.target_size, int):
                # If it's a single integer, make it a square
                self.target_size = (args.target_size, args.target_size)
            else:
                # Default to 256x256
                print(f"Warning: Invalid target_size format. Using default (256, 256).")
                self.target_size = (256, 256)
        else:
            self.target_size = (256, 256)
        
        if self.enable_full_image:
            print("Full image mode enabled: Images will keep their original resolution")
        else:
            print(f"Using target size: {self.target_size}")
        
        if self.challenge_mode:
            print("Challenge mode enabled: Filenames will be included in output!!!")
            print("Challenge mode enabled: No noise will be added to input images!!!")
        
        # Whether to convert to grayscale or keep RGB
        self.use_grayscale = args.use_grayscale if hasattr(args, 'use_grasyscale') else False
        
        # Find all image files in the directory and subdirectories
        self.img_list = []
        if hasattr(args, 'testdata_root'):
            # Walk through all subdirectories
            for root, _, files in os.walk(args.testdata_root):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        self.img_list.append(os.path.join(root, file))
            
            self.img_list = sorted(self.img_list)[:]
            print(f"Loading images from {args.testdata_root} (including subdirectories)")
        else:
            raise ValueError("testdata_root must be specified in args")
        
        # Optional pre-computed noisy images path
        self.noisy_img_list = None
        if hasattr(args, 'eval_noised_data_path') and args.eval_noised_data_path:
            # Similar recursive search for noisy images if path is provided
            self.noisy_img_list = []
            for root, _, files in os.walk(args.eval_noised_data_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        self.noisy_img_list.append(os.path.join(root, file))
            
            self.noisy_img_list = sorted(self.noisy_img_list)
        
        print(f'Found {len(self.img_list)} test images')
        print(f'Color mode: {"Grayscale" if self.use_grayscale else "Color (RGB)"}')
        
        # Optional: store image shape information for the first image
        # This helps with debugging but doesn't load all images
        try:
            sample_img = self.load_image(self.img_list[0])
            sample_img_tensor = self.preprocess_image(sample_img)
            print(f"Sample image shape: {sample_img_tensor.shape}")
        except Exception as e:
            print(f"Warning: Could not load sample image for shape check: {e}")
    
    def __len__(self):
        return len(self.img_list)
    
    def load_image(self, img_path):
        """Load an image from path"""
        try:
            # Load clean image
            orig_img = cv2.imread(img_path, 1)
            if orig_img is None:
                print(f"Warning: Could not read image {img_path}, using placeholder.")
                # Create a placeholder image of appropriate size
                if self.enable_full_image:
                    # Default placeholder size for full image mode
                    width, height = 512, 512
                else:
                    width, height = self.target_size
                return np.zeros((height, width, 3), dtype=np.uint8)
            
            # Only resize if enable_full_image is False
            if not self.enable_full_image:
                width, height = self.target_size
                orig_img = cv2.resize(orig_img, (width, height))
            
            # Convert from BGR to RGB (OpenCV loads as BGR)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            return orig_img
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            # Return a placeholder image
            if self.enable_full_image:
                # Default placeholder size for full image mode
                width, height = 512, 512
            else:
                width, height = self.target_size
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def preprocess_image(self, img):
        """Convert image to normalized tensor in proper format"""
        # Convert to PyTorch format (channels first)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        
        if self.use_grayscale:
            # Convert to grayscale by taking mean across channels
            img = np.mean(img, axis=0, keepdims=True)
        
        return img
    
    def add_noise(self, img):
        """Add noise to an image"""
        # Generate noisy version (working with floats in [0,1] range)
        noise = np.random.normal(0, self.sigma/255.0, img.shape)
        noisy_img = np.clip(img + noise, 0, 1.0).astype(np.float32)
        return noisy_img
    
    def __getitem__(self, idx):
        """
        Dynamically load and process an image when requested
        """
        # Get image path
        img_path = self.img_list[idx]
        
        # Load image
        img_raw = self.load_image(img_path)
        
        # Preprocess to normalized tensor format
        img = self.preprocess_image(img_raw)
        
        # Create sample structure based on mode
        if self.challenge_mode:
            # In challenge mode, the input images are already noisy and we don't have ground truth
            # So we treat the loaded image as the noisy input and create a dummy placeholder for labels
            # (the labels won't be used for evaluation in challenge mode)
            noisy_img = img  # Use the image as-is, no additional noise
            clean_img = img  # Create a dummy placeholder (won't be used)
        else:
            # In normal mode, we consider the loaded image as clean and add noise if needed
            clean_img = img
            
            # Get or create noisy version
            if self.noisy_img_list and idx < len(self.noisy_img_list):
                # Use pre-computed noisy image if available
                try:
                    noisy_img_raw = cv2.imread(self.noisy_img_list[idx], 1)
                    if noisy_img_raw is None:
                        # Fall back to generating noise
                        noisy_img = self.add_noise(clean_img)
                    else:
                        # Only resize if enable_full_image is False
                        if not self.enable_full_image:
                            width, height = self.target_size
                            noisy_img_raw = cv2.resize(noisy_img_raw, (width, height))
                        noisy_img_raw = cv2.cvtColor(noisy_img_raw, cv2.COLOR_BGR2RGB)
                        noisy_img = self.preprocess_image(noisy_img_raw)
                except Exception as e:
                    print(f"Error loading pre-computed noisy image, generating: {e}")
                    noisy_img = self.add_noise(clean_img)
            else:
                # Generate noisy version on the fly
                noisy_img = self.add_noise(clean_img)
        
        # Create the return dictionary
        sample = {
            'images': torch.from_numpy(noisy_img).float(),  # [C, H, W] format
            'labels': torch.from_numpy(clean_img).float(),  # [C, H, W] format
        }
        
        # Add filename if in challenge mode
        if self.challenge_mode:
            # Extract just the filename without path
            filename = os.path.basename(img_path)
            sample['name'] = filename
        
        return sample












import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class TrainSet_Dir(Dataset):
    """
    Memory-efficient training dataset with dynamic sampling:
    - Selects different samples for each training epoch without requiring explicit epoch setting
    - Maintains balanced folder selection
    - Adaptive resolution based on image size with three tiers: 896, 768, and 512
    """

    def __init__(self, args):
        super(TrainSet_Dir, self).__init__()

        # Parse parameters from args
        self.sigma = getattr(args, 'noise_sig', 25)         # Noise standard deviation
        self.patch_size = getattr(args, 'patch_size', 256)  # Default crop patch size
        self.phase = getattr(args, 'phase', 'train')        # 'train', 'valid', or 'test'
        self.args = args
        
        # New flag for adaptive resolution processing
        self.full = args.full if hasattr(args, 'full') else False
        
        # Set adaptive patch sizes when full mode is enabled - MODIFIED to 896, 768, 512
        self.large_patch_size = 896    # Use this for large images
        self.medium_patch_size = 768   # Use this for medium images
        self.small_patch_size = 512    # Fallback for smaller images
        
        if self.full:
            print(f"ADAPTIVE RESOLUTION MODE ENABLED: Using {self.large_patch_size}x{self.large_patch_size} crops for large images")
            print(f"ADAPTIVE RESOLUTION MODE ENABLED: Using {self.medium_patch_size}x{self.medium_patch_size} crops for medium images")
            print(f"ADAPTIVE RESOLUTION MODE ENABLED: Using {self.small_patch_size}x{self.small_patch_size} crops for smaller images")
        else:
            print(f"Patch-based training with patch size: {self.patch_size}x{self.patch_size}")

        # Organize images by folder for balanced selection
        self.folder_images = self._organize_images_by_folder(args.traindata_root)
        
        # Full list of all images for reference
        self.full_img_list = []
        for folder_imgs in self.folder_images.values():
            self.full_img_list.extend(folder_imgs)
        
        # Configure subset parameters
        self.use_subset = getattr(args, 'use_subset', True)  # Default to True for speed
        self.subset_fraction = getattr(args, 'train_subset_fraction', 1/32)  # Use 1/32 of data
        
        # Calculate total subset size based on fraction
        if self.use_subset:
            # Determine sample count per folder
            self.samples_per_folder = {}
            total_samples = 0
            
            for folder, images in self.folder_images.items():
                count = max(1, int(len(images) * self.subset_fraction))
                self.samples_per_folder[folder] = count
                total_samples += count
                
            self.subset_size = total_samples
        else:
            self.subset_size = len(self.full_img_list)
            
        # Track the current epoch for reproducible sampling
        self.epoch = 0
        
        # Print information about the dataset
        print(f'Found {len(self.full_img_list)} total training images across {len(self.folder_images)} folders')
        print(f'Will use approximately {self.subset_size} samples per epoch ({self.subset_fraction:.4f} of dataset)')
        
        # Optional: store image shape information for the first image
        try:
            sample_img = self.load_image(self.full_img_list[0])
            print(f"Sample image shape: {sample_img.shape}")
        except Exception as e:
            print(f"Warning: Could not load sample image for shape check: {e}")
        
        # Create a mapping of sample index to (folder, image_idx)
        # This will be dynamically recomputed for each epoch
        self._create_epoch_mapping()

    def _organize_images_by_folder(self, root_path):
        """
        Organize images by their parent folder for balanced selection.
        Returns a dictionary mapping folder names to lists of image paths.
        """
        folder_images = defaultdict(list)
        
        # Walk through all subdirectories
        for root, _, files in os.walk(root_path):
            # Skip the root directory itself
            if root == root_path:
                continue
                
            # Get the relative path from root
            rel_path = os.path.relpath(root, root_path)
            # Use first-level folder as the key
            folder_key = rel_path.split(os.sep)[0]
            
            # Add images from this folder
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    img_path = os.path.join(root, file)
                    folder_images[folder_key].append(img_path)
        
        # Sort each list for reproducibility
        for folder in folder_images:
            folder_images[folder] = sorted(folder_images[folder])
            print(f"Folder '{folder}': {len(folder_images[folder])} images")
            
        return dict(folder_images)  # Convert to regular dict
    
    def _create_epoch_mapping(self):
        """
        Create a mapping from dataset indices to actual image paths for the current epoch.
        This dynamically recreates the subset for each epoch.
        """
        # Set seed based on epoch for reproducibility within an epoch
        random.seed(42 + self.epoch)
        
        # Reset the mapping
        self.epoch_mapping = []
        
        if not self.use_subset:
            # Just use all images in order
            self.epoch_mapping = self.full_img_list[:]
            return
            
        # For each folder, select the predetermined number of samples
        for folder, images in self.folder_images.items():
            folder_subset_size = self.samples_per_folder[folder]
            
            # Randomly select images from this folder
            folder_samples = random.sample(images, folder_subset_size)
            
            # Add to the mapping
            self.epoch_mapping.extend(folder_samples)
        
        # Shuffle the entire mapping for better mixing
        random.shuffle(self.epoch_mapping)
        
        print(f'Epoch {self.epoch}: Generated new subset with {len(self.epoch_mapping)} samples')

    def set_epoch(self, epoch):
        """
        Update the current epoch and regenerate the sample mapping.
        Should be called at the beginning of each epoch.
        """
        if epoch != self.epoch:
            self.epoch = epoch
            self._create_epoch_mapping()

    def __len__(self):
        """Return the number of samples in the current subset"""
        return len(self.epoch_mapping)

    def load_image(self, img_path):
        """Load an image from path in RGB format."""
        try:
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"Warning: Could not read image {img_path}, using placeholder.")
                # Return a placeholder if image read fails
                return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            
            # Convert from BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)

    def random_crop(self, *images, patch_size):
        """
        Randomly crop a patch of size patch_size x patch_size from each image in `images`.
        Assumes all images in `images` have the same HxW shape.
        Returns original images if they are too small for the requested patch size.
        """
        h, w = images[0].shape[:2]
        if h < patch_size or w < patch_size:
            return images  # Return original images if they're too small
            
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        return [img[top:top+patch_size, left:left+patch_size, :] for img in images]

    def center_crop(self, *images, patch_size):
        """
        Center crop a patch of size patch_size x patch_size from each image in `images`.
        Assumes all images in `images` have the same HxW shape.
        Returns original images if they are too small for the requested patch size.
        """
        h, w = images[0].shape[:2]
        if h < patch_size or w < patch_size:
            return images  # Return original images if they're too small
            
        top = (h - patch_size) // 2
        left = (w - patch_size) // 2
        return [img[top:top+patch_size, left:left+patch_size, :] for img in images]

    def adaptive_crop(self, *images):
        """
        Apply adaptive cropping based on image size with three tiers:
        - 896x896 crop if image is large enough
        - 768x768 if image is medium sized
        - 512x512 if image is smaller
        """
        h, w = images[0].shape[:2]
        
        # Try large patch size first
        if h >= self.large_patch_size and w >= self.large_patch_size:
            if self.phase == 'train':
                return self.random_crop(*images, patch_size=self.large_patch_size)
            else:
                return self.center_crop(*images, patch_size=self.large_patch_size)
        
        # Try medium patch size
        elif h >= self.medium_patch_size and w >= self.medium_patch_size:
            if self.phase == 'train':
                return self.random_crop(*images, patch_size=self.medium_patch_size)
            else:
                return self.center_crop(*images, patch_size=self.medium_patch_size)
        
        # Fall back to smaller patch size
        elif h >= self.small_patch_size and w >= self.small_patch_size:
            if self.phase == 'train':
                return self.random_crop(*images, patch_size=self.small_patch_size)
            else:
                return self.center_crop(*images, patch_size=self.small_patch_size)
                
        # Image too small for any patching, just return original
        else:
            return images

    def augment(self, *images, hflip=True, rot=True):
        """
        Randomly apply horizontal flip, vertical flip, or 90-degree rotation to the images.
        The same augmentation is applied to all images in `images`.
        """
        # Decide which augmentations to apply
        do_hflip = hflip and (random.random() < 0.5)
        do_vflip = rot and (random.random() < 0.5)
        do_rot90 = rot and (random.random() < 0.5)

        def _augment(img):
            if do_hflip:
                img = img[:, ::-1, :]  # horizontal flip
            if do_vflip:
                img = img[::-1, :, :]  # vertical flip
            if do_rot90:
                img = img.transpose(1, 0, 2)  # rotate 90 degrees
            return img

        return [_augment(im) for im in images]

    def np2Tensor(self, *images):
        """
        Convert NumPy (H x W x C) arrays to PyTorch float tensors in (C x H x W).
        """
        def _to_tensor(img):
            # Transpose to CxHxW
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))
            return torch.from_numpy(img).float()

        return [_to_tensor(im) for im in images]

    def __getitem__(self, idx):
        """Get one sample (noisy image, clean image) from the dataset."""
        # Check if we need to generate new mapping for a new epoch
        # This assumes DataLoader uses Sequential sampler
        if idx == 0:
            # First sample of new data loader iteration - increment epoch
            self.epoch += 1
            self._create_epoch_mapping()
        
        # Get image path from the epoch mapping
        img_path = self.epoch_mapping[idx]

        # Load the clean (ground truth) image
        clean_img = self.load_image(img_path)

        # Optionally add noise
        clean_float = clean_img.astype(np.float32)
        noise = np.random.normal(0, self.sigma, clean_img.shape).astype(np.float32)
        noisy_img = np.clip(clean_float + noise, 0, 255).astype(np.uint8)

        if self.full:
            # Adaptive resolution mode - use large, medium or small patches based on image size
            clean_img, noisy_img = self.adaptive_crop(clean_img, noisy_img)
            
            # Apply data augmentation only during training
            if self.phase == 'train':
                clean_img, noisy_img = self.augment(clean_img, noisy_img)
        else:
            # Standard patch-based mode - crop patches of the default size
            if self.phase in ['train', 'valid']:
                clean_img, noisy_img = self.random_crop(clean_img, noisy_img, patch_size=self.patch_size)
                clean_img, noisy_img = self.augment(clean_img, noisy_img)
            else:
                # For test phase in patch mode, use center crop
                clean_img, noisy_img = self.center_crop(clean_img, noisy_img, patch_size=self.patch_size)

        # Convert from NumPy to PyTorch tensors
        clean_tensor, noisy_tensor = self.np2Tensor(clean_img, noisy_img)

        # Scale from [0..255] to [0..1]
        clean_tensor /= 255.0
        noisy_tensor /= 255.0

        # Return a dict with your keys
        sample = {
            'images': noisy_tensor,  # Noisy input
            'labels': clean_tensor   # Clean target
        }

        return sample


class TrainSet_MultiDir(Dataset):
    """
    Training dataset that combines images from multiple root directories.
    Handles both nested subdirectory structures (e.g., LSDIR) and flat directories (e.g., DIV2K).
    Supports per-epoch subset sampling with balanced folder selection.
    """

    def __init__(self, args):
        super(TrainSet_MultiDir, self).__init__()

        # Parse parameters from args
        self.sigma = getattr(args, 'noise_sig', 25)
        self.patch_size = getattr(args, 'patch_size', 256)
        self.phase = getattr(args, 'phase', 'train')
        self.args = args

        # Adaptive resolution settings
        self.full = args.full if hasattr(args, 'full') else False
        self.large_patch_size = 896
        self.medium_patch_size = 768
        self.small_patch_size = 512

        if self.full:
            print(f"ADAPTIVE RESOLUTION MODE ENABLED: {self.large_patch_size}/{self.medium_patch_size}/{self.small_patch_size}")
        else:
            print(f"Patch-based training with patch size: {self.patch_size}x{self.patch_size}")

        # Determine root directories
        roots = getattr(args, 'traindata_roots', None)
        if roots is None:
            roots = [args.traindata_root]
        print(f"TrainSet_MultiDir: scanning {len(roots)} root directories")

        # Collect images from all roots
        self.folder_images = {}
        for root_path in roots:
            root_path = root_path.rstrip(os.sep)
            dir_images = self._organize_images_from_dir(root_path)
            self.folder_images.update(dir_images)

        # Full list of all images
        self.full_img_list = []
        for folder_imgs in self.folder_images.values():
            self.full_img_list.extend(folder_imgs)

        # Subset configuration
        self.use_subset = getattr(args, 'use_subset', True)
        self.subset_fraction = getattr(args, 'train_subset_fraction', 1/32)

        if self.use_subset:
            self.samples_per_folder = {}
            total_samples = 0
            for folder, images in self.folder_images.items():
                count = max(1, int(len(images) * self.subset_fraction))
                self.samples_per_folder[folder] = count
                total_samples += count
            self.subset_size = total_samples
        else:
            self.subset_size = len(self.full_img_list)

        self.epoch = 0

        print(f'Found {len(self.full_img_list)} total training images across {len(self.folder_images)} folders')
        print(f'Will use approximately {self.subset_size} samples per epoch ({self.subset_fraction:.4f} of dataset)')

        try:
            sample_img = self.load_image(self.full_img_list[0])
            print(f"Sample image shape: {sample_img.shape}")
        except Exception as e:
            print(f"Warning: Could not load sample image for shape check: {e}")

        self._create_epoch_mapping()

    def _organize_images_from_dir(self, root_path):
        """
        Organize images from a single root directory.
        Handles both nested subdirectories and flat directories (root-level files).
        Prefixes folder keys with the root basename to avoid collisions across roots.
        """
        folder_images = defaultdict(list)
        root_basename = os.path.basename(root_path)

        for root, dirs, files in os.walk(root_path):
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            if not img_files:
                continue

            if root == root_path:
                # Root-level files -> use a special key
                folder_key = f"{root_basename}_root"
            else:
                rel_path = os.path.relpath(root, root_path)
                first_level = rel_path.split(os.sep)[0]
                folder_key = f"{root_basename}/{first_level}"

            for f in img_files:
                folder_images[folder_key].append(os.path.join(root, f))

        # Sort for reproducibility
        for folder in folder_images:
            folder_images[folder] = sorted(folder_images[folder])
            print(f"  Folder '{folder}': {len(folder_images[folder])} images")

        return dict(folder_images)

    def _create_epoch_mapping(self):
        """Create subset mapping for the current epoch."""
        random.seed(42 + self.epoch)
        self.epoch_mapping = []

        if not self.use_subset:
            self.epoch_mapping = self.full_img_list[:]
            return

        for folder, images in self.folder_images.items():
            folder_subset_size = self.samples_per_folder[folder]
            folder_samples = random.sample(images, folder_subset_size)
            self.epoch_mapping.extend(folder_samples)

        random.shuffle(self.epoch_mapping)
        print(f'Epoch {self.epoch}: Generated new subset with {len(self.epoch_mapping)} samples')

    def set_epoch(self, epoch):
        """Update epoch and regenerate mapping."""
        if epoch != self.epoch:
            self.epoch = epoch
            self._create_epoch_mapping()

    def __len__(self):
        return len(self.epoch_mapping)

    def load_image(self, img_path):
        """Load an image from path in RGB format."""
        try:
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"Warning: Could not read image {img_path}, using placeholder.")
                return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)

    def random_crop(self, *images, patch_size):
        h, w = images[0].shape[:2]
        if h < patch_size or w < patch_size:
            return images
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        return [img[top:top+patch_size, left:left+patch_size, :] for img in images]

    def center_crop(self, *images, patch_size):
        h, w = images[0].shape[:2]
        if h < patch_size or w < patch_size:
            return images
        top = (h - patch_size) // 2
        left = (w - patch_size) // 2
        return [img[top:top+patch_size, left:left+patch_size, :] for img in images]

    def adaptive_crop(self, *images):
        h, w = images[0].shape[:2]
        if h >= self.large_patch_size and w >= self.large_patch_size:
            if self.phase == 'train':
                return self.random_crop(*images, patch_size=self.large_patch_size)
            else:
                return self.center_crop(*images, patch_size=self.large_patch_size)
        elif h >= self.medium_patch_size and w >= self.medium_patch_size:
            if self.phase == 'train':
                return self.random_crop(*images, patch_size=self.medium_patch_size)
            else:
                return self.center_crop(*images, patch_size=self.medium_patch_size)
        elif h >= self.small_patch_size and w >= self.small_patch_size:
            if self.phase == 'train':
                return self.random_crop(*images, patch_size=self.small_patch_size)
            else:
                return self.center_crop(*images, patch_size=self.small_patch_size)
        else:
            return images

    def augment(self, *images, hflip=True, rot=True):
        do_hflip = hflip and (random.random() < 0.5)
        do_vflip = rot and (random.random() < 0.5)
        do_rot90 = rot and (random.random() < 0.5)

        def _augment(img):
            if do_hflip:
                img = img[:, ::-1, :]
            if do_vflip:
                img = img[::-1, :, :]
            if do_rot90:
                img = img.transpose(1, 0, 2)
            return img

        return [_augment(im) for im in images]

    def np2Tensor(self, *images):
        def _to_tensor(img):
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))
            return torch.from_numpy(img).float()
        return [_to_tensor(im) for im in images]

    def __getitem__(self, idx):
        if idx == 0:
            self.epoch += 1
            self._create_epoch_mapping()

        img_path = self.epoch_mapping[idx]
        clean_img = self.load_image(img_path)

        # Resize if image is too small for the patch size
        h, w = clean_img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            scale = max(self.patch_size / h, self.patch_size / w)
            new_h, new_w = int(h * scale) + 1, int(w * scale) + 1
            clean_img = cv2.resize(clean_img, (new_w, new_h))

        clean_float = clean_img.astype(np.float32)
        noise = np.random.normal(0, self.sigma, clean_img.shape).astype(np.float32)
        noisy_img = np.clip(clean_float + noise, 0, 255).astype(np.uint8)

        if self.full:
            clean_img, noisy_img = self.adaptive_crop(clean_img, noisy_img)
            if self.phase == 'train':
                clean_img, noisy_img = self.augment(clean_img, noisy_img)
        else:
            if self.phase in ['train', 'valid']:
                clean_img, noisy_img = self.random_crop(clean_img, noisy_img, patch_size=self.patch_size)
                clean_img, noisy_img = self.augment(clean_img, noisy_img)
            else:
                clean_img, noisy_img = self.center_crop(clean_img, noisy_img, patch_size=self.patch_size)

        clean_tensor, noisy_tensor = self.np2Tensor(clean_img, noisy_img)
        clean_tensor /= 255.0
        noisy_tensor /= 255.0

        return {
            'images': noisy_tensor,
            'labels': clean_tensor
        }


class TestSet_Dir(Dataset):
    """
    Memory-efficient test/validation dataset with dynamic sampling:
    - Generates a new balanced subset for each iteration through the dataset
    - Adaptive resolution based on image size with three tiers: 896, 768, and 512
    """
    def __init__(self, args):
        super(TestSet_Dir, self).__init__()
        
        # Parse parameters from args
        self.sigma = args.noise_sig if hasattr(args, 'noise_sig') else 50
        
        # Full resolution flag (can be provided through enable_full_image or full)
        self.full = (
            getattr(args, 'full', False) or 
            getattr(args, 'enable_full_image', False)
        )
        
        # Set adaptive patch sizes when full mode is enabled - MODIFIED to 896, 768, 512
        self.large_patch_size = 896    # Use this for large images
        self.medium_patch_size = 768   # Use this for medium images
        self.small_patch_size = 512    # Fallback for smaller images
        
        # Challenge mode flag
        self.challenge_mode = getattr(args, 'challenge', False)
        
        # Target size (used when full is False)
        if hasattr(args, 'target_size'):
            if isinstance(args.target_size, tuple) and len(args.target_size) == 2:
                self.target_size = args.target_size
            elif isinstance(args.target_size, int):
                # If it's a single integer, make it a square
                self.target_size = (args.target_size, args.target_size)
            else:
                # Default to 256x256
                print(f"Warning: Invalid target_size format. Using default (256, 256).")
                self.target_size = (256, 256)
        else:
            self.target_size = (256, 256)
        
        if self.full:
            print(f"ADAPTIVE RESOLUTION MODE ENABLED: Using {self.large_patch_size}x{self.large_patch_size} crops for large images")
            print(f"ADAPTIVE RESOLUTION MODE ENABLED: Using {self.medium_patch_size}x{self.medium_patch_size} crops for medium images")
            print(f"ADAPTIVE RESOLUTION MODE ENABLED: Using {self.small_patch_size}x{self.small_patch_size} crops for smaller images")
        else:
            print(f"Using target size: {self.target_size}")
        
        if self.challenge_mode:
            print("Challenge mode enabled: Filenames will be included in output!")
            print("Challenge mode enabled: No noise will be added to input images!")
        
        # Whether to convert to grayscale or keep RGB
        self.use_grayscale = getattr(args, 'use_grayscale', False)
        
        # Organize images by folder for balanced selection
        self.folder_images = self._organize_images_by_folder(args.testdata_root)
        
        # Full list of all images for reference
        self.full_img_list = []
        for folder_imgs in self.folder_images.values():
            self.full_img_list.extend(folder_imgs)
        
        # Optional pre-computed noisy images
        self.noisy_image_dict = {}
        if hasattr(args, 'eval_noised_data_path') and args.eval_noised_data_path:
            self._load_noisy_images(args.eval_noised_data_path)
            
        # Configure subset parameters
        self.use_subset = getattr(args, 'use_subset', True)  # Default to True for speed
        self.subset_fraction = getattr(args, 'test_subset_fraction', 1/5)  # Use 1/5 of data
        
        # Calculate subset size for each folder
        if self.use_subset:
            # Determine sample count per folder
            self.samples_per_folder = {}
            total_samples = 0
            
            for folder, images in self.folder_images.items():
                count = max(1, int(len(images) * self.subset_fraction))
                self.samples_per_folder[folder] = count
                total_samples += count
                
            self.subset_size = total_samples
        else:
            self.subset_size = len(self.full_img_list)
        
        # Track the current evaluation run for reproducible sampling
        self.eval_run = 0
        
        # Create initial mapping
        self._create_run_mapping()
        
        # Print information about the dataset
        print(f'Found {len(self.full_img_list)} total test images across {len(self.folder_images)} folders')
        print(f'Will use approximately {self.subset_size} samples per evaluation ({self.subset_fraction:.4f} of dataset)')
        print(f'Color mode: {"Grayscale" if self.use_grayscale else "Color (RGB)"}')
        
        # Optional: store image shape information for the first image
        try:
            sample_img = self.load_image(self.full_img_list[0])
            sample_img_tensor = self.preprocess_image(sample_img)
            print(f"Sample image shape: {sample_img_tensor.shape}")
        except Exception as e:
            print(f"Warning: Could not load sample image for shape check: {e}")
    
    def _organize_images_by_folder(self, root_path):
        """
        Organize images by their parent folder for balanced selection.
        Returns a dictionary mapping folder names to lists of image paths.
        """
        folder_images = defaultdict(list)
        
        # Walk through all subdirectories
        for root, _, files in os.walk(root_path):
            # Skip the root directory itself
            if root == root_path:
                continue
                
            # Get the relative path from root
            rel_path = os.path.relpath(root, root_path)
            # Use first-level folder as the key
            folder_key = rel_path.split(os.sep)[0]
            
            # Add images from this folder
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    img_path = os.path.join(root, file)
                    folder_images[folder_key].append(img_path)
        
        # Sort each list for reproducibility
        for folder in folder_images:
            folder_images[folder] = sorted(folder_images[folder])
            print(f"Folder '{folder}': {len(folder_images[folder])} images")
            
        return dict(folder_images)  # Convert to regular dict
    
    def _load_noisy_images(self, noisy_root_path):
        """Load noisy images and create a lookup dictionary by filename"""
        self.noisy_image_dict = {}
        
        # Walk through all subdirectories in the noisy images path
        for root, _, files in os.walk(noisy_root_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    # Use the filename as the key
                    self.noisy_image_dict[file] = os.path.join(root, file)
        
        print(f"Found {len(self.noisy_image_dict)} pre-computed noisy images for evaluation")
    
    def _create_run_mapping(self):
        """
        Create a mapping from dataset indices to actual image paths for the current evaluation run.
        This dynamically recreates the subset for each run.
        """
        # Set seed based on run for reproducibility within a run
        random.seed(100 + self.eval_run)
        
        # Reset the mapping
        self.run_mapping = []
        
        if not self.use_subset:
            # Just use all images in order
            self.run_mapping = self.full_img_list[:]
            return
            
        # For each folder, select the predetermined number of samples
        for folder, images in self.folder_images.items():
            folder_subset_size = self.samples_per_folder[folder]
            
            # Randomly select images from this folder
            folder_samples = random.sample(images, folder_subset_size)
            
            # Add to the mapping
            self.run_mapping.extend(folder_samples)
        
        # Shuffle the entire mapping for better mixing
        random.shuffle(self.run_mapping)
        
        print(f'Evaluation Run {self.eval_run}: Generated new subset with {len(self.run_mapping)} samples')
    
    def set_eval_run(self, run_id):
        """
        Update the current evaluation run and regenerate the sample mapping.
        """
        if run_id != self.eval_run:
            self.eval_run = run_id
            self._create_run_mapping()
    
    def __len__(self):
        """Return the number of samples in the current subset"""
        return len(self.run_mapping)
    
    def center_crop(self, img, patch_size):
        """
        Center crop a patch of size patch_size x patch_size from an image.
        Returns original image if it's too small for the requested patch size.
        """
        h, w = img.shape[:2]
        if h < patch_size or w < patch_size:
            return img  # Return original image if it's too small
            
        top = (h - patch_size) // 2
        left = (w - patch_size) // 2
        return img[top:top+patch_size, left:left+patch_size, :]

    def adaptive_crop(self, img):
        """
        Apply adaptive cropping based on image size with three tiers:
        - 896x896 crop if image is large enough
        - 768x768 if image is medium sized
        - 512x512 if image is smaller
        """
        h, w = img.shape[:2]
        
        # Try large patch size first
        if h >= self.large_patch_size and w >= self.large_patch_size:
            return self.center_crop(img, self.large_patch_size)
        
        # Try medium patch size
        elif h >= self.medium_patch_size and w >= self.medium_patch_size:
            return self.center_crop(img, self.medium_patch_size)
        
        # Fall back to smaller patch size
        elif h >= self.small_patch_size and w >= self.small_patch_size:
            return self.center_crop(img, self.small_patch_size)
                
        # Image too small for any patching, just return original
        else:
            return img
    
    def load_image(self, img_path):
        """Load an image from path"""
        try:
            # Load clean image
            orig_img = cv2.imread(img_path, 1)
            if orig_img is None:
                print(f"Warning: Could not read image {img_path}, using placeholder.")
                # Create a placeholder image of appropriate size
                if self.full:
                    # Default placeholder size for full image mode
                    width, height = 512, 512
                else:
                    width, height = self.target_size
                return np.zeros((height, width, 3), dtype=np.uint8)
            
            # Process image according to mode
            if not self.full:
                # Resize to target size
                width, height = self.target_size
                orig_img = cv2.resize(orig_img, (width, height))
            
            # Convert from BGR to RGB (OpenCV loads as BGR)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            return orig_img
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            # Return a placeholder image
            if self.full:
                # Default placeholder size for full image mode
                width, height = 512, 512
            else:
                width, height = self.target_size
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def preprocess_image(self, img):
        """Convert image to normalized tensor in proper format"""
        # Convert to PyTorch format (channels first)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        
        if self.use_grayscale:
            # Convert to grayscale by taking mean across channels
            img = np.mean(img, axis=0, keepdims=True)
        
        return img
    
    def add_noise(self, img):
        """Add noise to an image"""
        # Generate noisy version (working with floats in [0,1] range)
        noise = np.random.normal(0, self.sigma/255.0, img.shape)
        noisy_img = np.clip(img + noise, 0, 1.0).astype(np.float32)
        return noisy_img
    
    def __getitem__(self, idx):
        """
        Dynamically load and process an image when requested
        """
        # Check if we need to generate new mapping for a new evaluation
        # This assumes DataLoader uses Sequential sampler
        if idx == 0:
            # First sample of new data loader iteration - increment run
            self.eval_run += 1
            self._create_run_mapping()
        
        # Get image path from the run mapping
        img_path = self.run_mapping[idx]
        
        # Get filename for matching with noisy images and challenge mode
        filename = os.path.basename(img_path)
        
        # Load image
        img_raw = self.load_image(img_path)
        
        # Apply adaptive cropping if in full mode
        if self.full:
            img_raw = self.adaptive_crop(img_raw)
            
        # Preprocess to normalized tensor format
        img = self.preprocess_image(img_raw)
        
        # Create sample structure based on mode
        if self.challenge_mode:
            # In challenge mode, the input images are already noisy and we don't have ground truth
            # So we treat the loaded image as the noisy input and create a dummy placeholder for labels
            noisy_img = img  # Use the image as-is, no additional noise
            clean_img = img  # Create a dummy placeholder (won't be used)
            
            # Create sample with the filename for challenge mode
            sample = {
                'images': torch.from_numpy(noisy_img).float(),  # [C, H, W] format
                'labels': torch.from_numpy(clean_img).float(),  # [C, H, W] format
                'name': filename                                # Include filename for challenge mode
            }
        else:
            # In normal mode, we consider the loaded image as clean and add noise if needed
            clean_img = img
            
            # Check if we have a matching pre-computed noisy image
            if filename in self.noisy_image_dict:
                try:
                    # Use pre-computed noisy image
                    noisy_img_path = self.noisy_image_dict[filename]
                    noisy_img_raw = cv2.imread(noisy_img_path, 1)
                    
                    if noisy_img_raw is None:
                        # Fall back to generating noise
                        noisy_img = self.add_noise(clean_img)
                    else:
                        # Process according to mode
                        if not self.full:
                            width, height = self.target_size
                            noisy_img_raw = cv2.resize(noisy_img_raw, (width, height))
                        else:
                            # Crop to same size as clean image if in full mode
                            noisy_img_raw = self.adaptive_crop(noisy_img_raw)
                            
                        noisy_img_raw = cv2.cvtColor(noisy_img_raw, cv2.COLOR_BGR2RGB)
                        noisy_img = self.preprocess_image(noisy_img_raw)
                except Exception as e:
                    print(f"Error loading pre-computed noisy image, generating: {e}")
                    noisy_img = self.add_noise(clean_img)
            else:
                # Generate noisy version on the fly
                noisy_img = self.add_noise(clean_img)
            
            # Create sample without filename for non-challenge mode
            sample = {
                'images': torch.from_numpy(noisy_img).float(),  # [C, H, W] format
                'labels': torch.from_numpy(clean_img).float(),  # [C, H, W] format
            }

        return sample

# Import MEF dataset classes so they can be found by importlib
from .dataset_mef import TrainSet_MEF, TestSet_MEF