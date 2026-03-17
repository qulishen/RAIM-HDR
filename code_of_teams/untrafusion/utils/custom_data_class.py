import torch
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        super(CustomDataset, self).__init__()
        # Initialize dataset properties here
        self.root_dir = root_dir
        self.transform = transform
        
        # Load your dataset files or directories here
        self.files = []  # Replace this with actual file loading logic
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Returns:
            inputs (Tensor): A batch of 8 input images.
            target (Tensor): The corresponding ground truth image.
        """
        # Load the input and target images from disk or memory
        input_images = []
        for i in range(8):  # Assuming you have 8 input images per sample
            img_path = f"{self.root_dir}/input_{i+1}.png"  # Adjust path according to your naming convention
            img_tensor = Image.open(img_path).convert('RGB')
            if self.transform:
                img_tensor = self.transform(img_tensor)
            input_images.append(img_tensor)
        
        target_img_path = f"{self.root_dir}/target.png"  # Adjust path according to your naming convention
        target_tensor = Image.open(target_img_path).convert('RGB')
        if self.transform:
            target_tensor = self.transform(target_tensor)
        
        inputs = torch.stack(input_images)  # Stack the input images into a single tensor
        target = target_tensor
        
        return inputs, target