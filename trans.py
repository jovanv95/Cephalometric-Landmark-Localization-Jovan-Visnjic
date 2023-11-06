
import pandas as pd
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import imageio
import random
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms


def transformer_train(image, x, y, params=True, p=0.8, target_size=(256, 256), mean=1, std=1):
    # Convert image to numpy array
    image = np.asanyarray(image)
    image = imageio.core.util.Array(image)

    # Randomly adjust image size
    rand_num_h = random.uniform(0.75, 1)
    rand_num_w = random.uniform(0.75, 1)
    small_h = int(256 * rand_num_h)
    small_w = int(256 * rand_num_w)

    # Apply image transformations using imgaug
    seq1 = iaa.Sequential([
        iaa.Sometimes(0.4, iaa.ChangeColorTemperature((2000, 6000))),
        iaa.Resize({"height": target_size[0], "width": target_size[1]}),
        iaa.Resize({"height": small_h, "width": small_w}),
        iaa.Sometimes(1, iaa.Affine(rotate=(random.uniform(-1, 1) * 15))),
        iaa.PadToFixedSize(height=target_size[0], width=target_size[1])
    ])

    # Create KeypointsOnImage object
    kps = KeypointsOnImage([Keypoint(x=x[i], y=y[i]) for i in range(22)], shape=image.shape)

    # Apply image augmentation and get augmented keypoints
    image_aug, kps_aug = seq1(image=image, keypoints=kps)

    sigma = 1
    heatmaps = []
    keypoints = []

    for key_point in kps_aug:
        keypoint = [key_point.x, key_point.y]
        keypoints.append(torch.tensor(keypoint))
        heatmap_size = (256, 256)
        keypoint_x, keypoint_y = key_point.x, key_point.y

        x = torch.arange(heatmap_size[0])
        y = torch.arange(heatmap_size[1])
        xx, yy = torch.meshgrid(x, y)

        # Generate Gaussian heatmap for each keypoint
        gaussian = torch.exp(-((xx - keypoint_x)**2 + (yy - keypoint_y)**2) / (2 * sigma**2))
        heatmap = gaussian / torch.max(gaussian)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmaps.append(heatmap)

    # Combine the heatmaps into a single tensor
    heatmap = torch.cat(heatmaps, dim=1)
    heatmap = heatmap.squeeze(0)

    # Replace zeros in the heatmap with -1 and scale the values
    heatmap = torch.where(heatmap == 0, torch.tensor(-1), heatmap)
    heatmap = torch.mul(heatmap, 10)
    keypoints = torch.stack(keypoints)

    # Prepare the image for the model
    image = np.array(image_aug)
    image = torch.tensor(image)
    image = image.to(torch.float32)

    # Normalize the image and change its dimension order
    normalize = transforms.Normalize(mean=mean, std=std)
    image = normalize(image)
    image = image.permute(2, 0, 1)

    return image, heatmap


def transformer_test(image, x, y, params=True, p=0.25, target_size=(256, 256), mean=None, std=None):
    # Convert image to NumPy array
    image = np.asarray(image)
    image = imageio.core.util.Array(image)

    # Create an image resizing sequence
    seq2 = iaa.Sequential([
        iaa.Resize({"height": target_size[0], "width": target_size[1]}),
    ])

    # Create KeypointsOnImage from x and y coordinates
    kps = KeypointsOnImage([Keypoint(x=x[i], y=y[i]) for i in range(22)], shape=image.shape)

    # Apply the resizing sequence to the image and keypoints
    image_aug, kps_aug = seq2(image=image, keypoints=kps)

    sigma = 1
    heatmaps = []
    keypoints = []

    # Generate heatmaps for each key point
    for key_point in kps_aug:
        keypoint = [key_point.x, key_point.y]
        keypoints.append(torch.tensor(keypoint))
        heatmap_size = (256, 256)
        keypoint_x, keypoint_y = key_point.x, key_point.y
        x = torch.arange(heatmap_size[0])
        y = torch.arange(heatmap_size[1])
        xx, yy = torch.meshgrid(x, y)
        gaussian = torch.exp(-((xx - keypoint_x) ** 2 + (yy - keypoint_y) ** 2) / (2 * sigma ** 2))
        heatmap = gaussian / torch.max(gaussian)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmaps.append(heatmap)

    # Concatenate heatmaps along the channel dimension
    heatmap = torch.cat(heatmaps, dim=1)
    heatmap = heatmap.squeeze(0)
    heatmap = torch.where(heatmap == 0, torch.tensor(-1), heatmap)
    heatmap = torch.mul(heatmap, 10)
    keypoints = torch.stack(keypoints)

    # Convert the augmented image to a PyTorch tensor
    image = np.array(image_aug)
    image = torch.tensor(image)
    image = image.to(torch.float32)

    # Normalize the image using mean and standard deviation
    normalize = transforms.Normalize(mean=mean, std=std)
    image = normalize(image)

    # Permute the image dimensions to (C, H, W) format
    image = image.permute(2, 0, 1)

    return image, heatmap







def transformer_mean_std(image,x,y,params = True,p = 0.25, target_size = (256,256),mean = None,std = None):

    image = np.asanyarray(image)
    image = imageio.core.util.Array(image)
    seq = iaa.Sequential([
        iaa.Resize({
            "height": target_size[0], 
            "width": target_size[1]
            })])


    image_aug = seq(image=image)


    image = np.array(image_aug)
    image = torch.tensor(image)
    image = image.to(torch.float32)


    return image , image
