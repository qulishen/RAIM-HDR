# import torch
# from torchvision import datasets
# import torchvision.transforms as transforms
# import os
# from PIL import Image
# import cv2

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, transform=transforms.ToTensor(), train=True):
#         super(CustomDataset, self).__init__()
#         # Initialize dataset properties here
#         self.root_dir = root_dir
#         self.transform = transform
#         self.train = train
        
#         # Load your dataset files or directories here
#         self.files = os.listdir(root_dir)  # List all files in the directory
#         print(f"Loaded {len(self.files)} files.")  # Replace this with actual file loading logic
    
#     def __len__(self):
#         return len(self.files) // 10    # input + gt = 10 frames per scene
    
#     def __getitem__(self, idx):
#         """
#         Returns:
#             inputs (Tensor): A batch of 9 input images.
#             target (Tensor): The corresponding ground truth image.
#         """
#         # Load the input and target images from disk or memory
#         input_images = []

#         if not self.train:
#             idx = idx * 10
#             names = os.listdir(self.root_dir)
#             for i in range(9):  # Assuming you have 8 input images per sample
#                 img_path = f"{self.root_dir}{names[idx+i+1]}"  # Adjust path according to your naming convention
#                 img_tensor = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#                 if self.transform:
#                     img_tensor = self.transform(img_tensor)
#                 input_images.append(img_tensor)
            
#             target_img_path = f"{self.root_dir}{names[idx]}"  # Adjust path according to your naming convention

#             target_tensor = cv2.imread(target_img_path, cv2.IMREAD_UNCHANGED)
#             if self.transform:
#                 target_tensor = self.transform(target_tensor)
#         else:
#             for i in range(9):  # Assuming you have 8 input images per sample
#                 img_path = f"{self.root_dir}Scene-{idx:03}-in-{i}.tif"  # Adjust path according to your naming convention
#                 img_tensor = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#                 if self.transform:
#                     img_tensor = self.transform(img_tensor)
#                 input_images.append(img_tensor)
        
#             target_img_path = f"{self.root_dir}Scene-{idx:03}-gt.tif"  # Adjust path according to your naming convention

#             target_tensor = cv2.imread(target_img_path, cv2.IMREAD_UNCHANGED)
#             if self.transform:
#                 target_tensor = self.transform(target_tensor)
        
#         inputs = torch.stack(input_images)  # Stack the input images into a single tensor
#         target = target_tensor
        
#         return inputs, target

import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        super(CustomDataset, self).__init__()
        self.root_dir = os.path.join(root_dir)
        self.transform = transform
        self.train = train
        
        # 获取所有场景文件夹（如 000, 001, 002...）
        # 排除非文件夹的文件
        self.scenes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        
        print(f"Loaded {len(self.scenes)} scenes from {root_dir}.")

        if len(self.scenes) > 0:
            first_scene_path = os.path.join(self.root_dir, self.scenes[0])
            self.input_num = len([f for f in os.listdir(first_scene_path) if f.lower().endswith('.jpg')])
        else:
            self.input_num = 0

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.root_dir, scene_name)
        
        input_images = []
        loop_range = self.input_num - 1 if self.train else self.input_num
        
        # 1. 读取 self.input_num-1 帧输入 (7帧情形: 0.jpg 到 6.jpg)
        for i in range(0, loop_range): 
            img_path = os.path.join(scene_path, f"{i}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img) # 转换为 Tensor [3, H, W]
            input_images.append(img)
        
        # 2. 读取 Ground Truth (HDR.jpg)
        if self.train:
            target_path = os.path.join(scene_path, "HDR.jpg") 
            target_img = cv2.imread(target_path)
            if target_img is None:
                raise FileNotFoundError(f"无法读取真值图片: {target_path}")
             
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            if self.transform:
                target_tensor = self.transform(target_img)
        else:
            target_tensor = torch.zeros(1)
            
        # 将输入序列堆叠为 [7, 3, H, W]
        inputs = torch.stack(input_images)
        target = target_tensor

        # --- 关键修改：实时数据增强 (仅在训练模式开启) ---
        if self.train:
            # 1. 随机打乱帧顺序（固定参考帧）
            ref_frame_idx = 3  # 参考帧索引，保持不动
            if inputs.shape[0] > 2:  # 至少需要3帧才有打乱的意义
                # 获取参考帧
                ref_frame = inputs[ref_frame_idx].clone()

                # 获取其他帧的索引（不包括参考帧）
                other_indices = [i for i in range(inputs.shape[0]) if i != ref_frame_idx]

                # 随机打乱其他帧
                shuffled_indices = other_indices.copy()
                np.random.shuffle(shuffled_indices)

                # 重建索引列表：参考帧位置保持不变，其他帧随机打乱
                new_order = []
                for i in range(inputs.shape[0]):
                    if i == ref_frame_idx:
                        new_order.append(ref_frame_idx)
                    else:
                        new_order.append(shuffled_indices.pop(0))

                inputs = inputs[new_order]

            # 2. 随机水平翻转
            if np.random.random() > 0.5:
                inputs = torch.flip(inputs, dims=[3]) # [7, 3, H, W] 的最后维度是 W
                target = torch.flip(target, dims=[2]) # [3, H, W] 的最后维度是 W

            # 3. 随机垂直翻转
            if np.random.random() > 0.5:
                inputs = torch.flip(inputs, dims=[2]) # H 维度
                target = torch.flip(target, dims=[1]) # H 维度

            # 4. 随机旋转 (90, 180, 270度)
            k = np.random.randint(0, 4) # 随机旋转次数
            if k > 0:
                inputs = torch.rot90(inputs, k, dims=[2, 3])
                target = torch.rot90(target, k, dims=[1, 2])

        return inputs, target