from torch.utils.data import DataLoader
from ffhqsub_dataset import FFHQsubDataset
import yaml
from torch.utils.data import DataLoader
from ffhqsub_dataset import FFHQsubDataset
import yaml
import matplotlib.pyplot as plt


# Load the parameters from the yml file
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as utils

# Load the parameters from the yml file
with open('./src/traintest123.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Create the dataset
dataset = FFHQsubDataset(params)
# Use DataLoader to handle batching
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch in dataloader:
    print(batch['gt'].shape)
    # print(batch['lq'].shape)
    # print(batch.keys())
    
    # plt.imshow(batch['gt'][0].permute(1, 2, 0).numpy())
    # plt.imshow(batch["kernel1"][0].permute(1, 2, 0).numpy())
    # plt.imshow(batch["kernel2"][0].permute(1, 2, 0).numpy())
    # plt.imshow(batch["sinc_kernel"][0].permute(1, 2, 0).numpy())
    # plt.show()

    break

# # Check the shape of the tensor
# print(batch["kernel1"][0].shape)

# # If it's a 2D tensor but needs to be treated as a single-channel image:
# tensor = batch["kernel1"][0].unsqueeze(0)  # Add a channel dimension
# tensor = tensor.permute(1, 2, 0)  # Adjust dimensions for plotting
# plt.imshow(tensor.numpy(), cmap='gray')  # Use a colormap for single-channel
# plt.savefig('kernel1output.png')

# # Check the shape of the tensor
# print(batch["kernel2"][0].shape)

# # If it's a 2D tensor but needs to be treated as a single-channel image:
# tensor = batch["kernel2"][0].unsqueeze(0)  # Add a channel dimension
# tensor = tensor.permute(1, 2, 0)  # Adjust dimensions for plotting
# plt.imshow(tensor.numpy(), cmap='gray')  # Use a colormap for single-channel
# plt.savefig('kernel2output.png')

# # Check the shape of the tensor
# print(batch["sinc_kernel"][0].shape)

# # If it's a 2D tensor but needs to be treated as a single-channel image:
# tensor = batch["sinc_kernel"][0].unsqueeze(0)  # Add a channel dimension
# tensor = tensor.permute(1, 2, 0)  # Adjust dimensions for plotting
# plt.imshow(tensor.numpy(), cmap='gray')  # Use a colormap for single-channel
# plt.savefig('sinc_kerneloutput.png')

import torch.nn.functional as F
import torch 

def apply_kernel_and_downsample(img_tensor, kernel, scale_factor=0.5):
    C, H, W = img_tensor.shape
    kernel = kernel.repeat(C, 1, 1, 1)  # Repeat kernel C times
    padding = kernel.size(2) // 2
    blurred_img = F.conv2d(img_tensor.unsqueeze(0), kernel, padding=padding, groups=C).squeeze(0)
    
    # Downsample image using bicubic interpolation
    height, width = int(H * scale_factor), int(W * scale_factor)
    downsampled_img = F.interpolate(blurred_img.unsqueeze(0), size=(height, width), mode='bicubic', align_corners=False).squeeze(0)
    return downsampled_img

def apply_kernel_and_downsample(img_tensor, kernel, scale_factor=0.5):
    C, H, W = img_tensor.shape
    print("Original max:", img_tensor.max(), "Original min:", img_tensor.min())
    
    kernel = kernel.repeat(C, 1, 1, 1)
    padding = kernel.size(2) // 2
    blurred_img = F.conv2d(img_tensor.unsqueeze(0), kernel, padding=padding, groups=C).squeeze(0)
    
    print("Blurred max:", blurred_img.max(), "Blurred min:", blurred_img.min())
    
    blurred_img = torch.clamp(blurred_img, 0, 1)
    height, width = int(H * scale_factor), int(W * scale_factor)
    downsampled_img = F.interpolate(blurred_img.unsqueeze(0), size=(height, width), mode='bicubic', align_corners=False).squeeze(0)
    
    print("Downsampled max:", downsampled_img.max(), "Downsampled min:", downsampled_img.min())
    
    downsampled_img = torch.clamp(downsampled_img, 0, 1)
    
    return downsampled_img

# Example usage
scale_factor = 0.25  # Adjust scale factor as needed
for batch in dataloader:
    gt_images = batch['gt']     # Ground truth images
    kernels1 = batch['kernel1'] # First set of kernels
    blurred_images = [apply_kernel_and_downsample(img, kern, scale_factor) for img, kern in zip(batch['gt'], batch['kernel1'])]

    for i, img in enumerate(blurred_images):
        plt.figure()
        img_display = img.permute(1, 2, 0).cpu().detach().numpy()  # Convert CHW to HWC for visualization
        # plt.imshow(img_display)
        plt.title(f'Blurred and Downsampled Image {i+1}')
        plt.savefig(f'blurred_image_{i+1}.png')
        plt.close()
        break