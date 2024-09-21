## Use this file only for feature extraction of ViT patch full images
import sys

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models import ViT_B_16_Weights
import torchvision.transforms as transforms
from PIL import Image as PIL_Image
from patch_ablation_utils import PatchMaster
import vision_transformer
from tqdm import tqdm
from pathlib import Path


import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument("-dataset_common_objects_df_path", "--dataset_common_objects_df_path", type=str)
parser.add_argument("-img_dir", "--img_dir",type=str)
parser.add_argument("-no_masks_dir", "--no_masks_dir",type=str)
parser.add_argument("-save_path", "--save_path", type=str)
parser.add_argument("-model_name", "--model_name", type=str, default="vit_base_patch16_224", help="Model name, e.g., 'vit_base_patch16_224' or 'vit_large_patch32_384'")
args = parser.parse_args()

class CenterCropLongEdge():
    """
    https://github.com/facebookresearch/DIG-In/blob/main/utils.py#L21
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))
    
    def __repr__(self) -> str:
        return self.__class__.__name__

TRANSFORM_NORMALIZE = torchvision.transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

pm = PatchMaster("vit")

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def mask_image(image, patch_mask, filler_val= 0, n_patches=14, patch_size=16):
    c,h,w = image.shape
    img = torch.ones(image.shape)*filler_val
    
    for i in patch_mask:
        if i>0: 
            x = (i-1)//n_patches
            y = (i-1)%n_patches
            img[:, x*patch_size: (x+1)*patch_size, y*patch_size: (y+1)*patch_size] = image[:, x*patch_size: (x+1)*patch_size, y*patch_size: (y+1)*patch_size]

    return img

def get_attention_masks(patches, device="cuda"):
    # patches: batch x (n_patch*n_patch+1)
    # return torch.hstack((torch.ones(patches.shape[0],1).to(device),torch.any(patches!=0, dim=-1)))
    return torch.any(patches!=0, dim=-1)

def generate_features(df_path, img_dir, no_masks_dir, save_path, model_name="vit_base_patch16_224"):
    if model_name == "vit_base_patch16_224":
        transformer = vision_transformer.vit_base_patch16_224(pretrained=True).to("cuda")
        img_size, patch_size = 224, 16
    elif model_name == "vit_base_patch32_384":
        transformer = vision_transformer.vit_base_patch32_384(pretrained=True).to("cuda")
        img_size, patch_size = 384, 32
    elif model_name == "vit_large_patch16_224":
        transformer = vision_transformer.vit_large_patch16_224(pretrained=True).to("cuda")
        img_size, patch_size = 224, 16
    elif model_name == "vit_large_patch32_384":
        transformer = vision_transformer.vit_large_patch32_384(pretrained=True).to("cuda")
        img_size, patch_size = 384, 32
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    n_patches = img_size // patch_size
    
    transforms_list = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize(img_size),  # Use dynamic img_size based on the model
        transforms.ToTensor(),
        # TRANSFORM_NORMALIZE
    ])
    
    preprocessor1 = transforms_list
    preprocessor2 = TRANSFORM_NORMALIZE
    # reading the data
    df = pd.read_csv(df_path)

    features = []
    is_good_features = []
    masked = []
    for _,row in tqdm(df.iterrows()):
        img_path = Path(f"{img_dir}/{row['file_path']}")
        if img_path.is_file():
            try: 
                with torch.no_grad():
                    image = PIL_Image.open(img_path)
                    image = preprocessor1(image) # resizing
                    patches = patchify(image.unsqueeze(0), n_patches)
                    mask_mod = get_attention_masks(patches, "cpu")
                    ordering = torch.hstack(((mask_mod[0]==0).nonzero()[:,0], mask_mod[0].nonzero()[:,0]))
                    ordering = ordering.unsqueeze(0).repeat(3,1)
                    nums_to_exclude = (mask_mod[0]==0).nonzero()[:,0].shape[0]
                    patch_mask = pm.get_patch_masks(ordering, nums_to_exclude)

                    image = preprocessor2(image)
                    # tensor_image = preprocess(image).to("cuda").float()
                    image = image.cuda()
                    patch_mask = patch_mask[0,:].unsqueeze(0).cuda()
                    feature = transformer.forward_features(image.unsqueeze(0), patch_mask)[0].cpu().detach().numpy()
                    features.append(
                        feature
                    )
                    is_good_features.append(1)
                    masked.append(1)

            except Exception as e:
                print(e)
                features.append([])
                is_good_features.append(0)
                masked.append(0)
        else:
            try:
                img_path = f"{no_masks_dir}/{row['file_path']}"
                print(f"{img_path} in no masks dir")
                with torch.no_grad():
                    # tensor_image = preprocess(image).to("cuda").float()
                    features.append(
                        []
                    )
                    is_good_features.append(1)
                    masked.append(0)
            except Exception as e:
                print(e)
                features.append([])
                is_good_features.append(0)
                masked.append(0)


    df["features"] = features
    df["is_good"] = is_good_features
    df["masked"] = masked
    print(f"feature extraction done for {sum(is_good_features)} out of {len(is_good_features)}")
    df = df[df["is_good"]==1]

    df.to_pickle(save_path)
    

# if __name__=="__main__":
#     generate_features(args.dataset_common_objects_df_path, args.img_dir, args.no_masks_dir, args.save_path, args.model_name)
