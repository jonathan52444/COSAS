import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry

from torchvision import transforms
import SimpleITK as sitk
import SimpleITK
from scipy.ndimage import zoom
# from icecream import ic

def read(path):
    image = SimpleITK.ReadImage(path)
    return SimpleITK.GetArrayFromImage(image)

def main()
    # Define default variables
    config = None  # The config file provided by the trained model
    input_root = '/input/images/adenocarcinoma-image'
    output_root = '/output/images/adenocarcinoma-mask'
    dataset = 'COSAS'  # Experiment name
    num_classes = 2
    list_dir = '/cluster/project7/SAMed/samed_codes/SAMed/lists/lists_Synapse/'
    img_size = 512  # Input image size of the network
    input_size = 256  # The input size for training SAM model
    seed = 1234  # Random seed
    is_savenii = False  # Whether to save results during inference
    deterministic = 1  # Whether to use deterministic training
    ckpt = 'sam_vit_b_01ec64.pth'  # Pretrained checkpoint
    lora_ckpt = 'checkpoint_199_512.pth'  # The checkpoint from LoRA
    vit_name = 'vit_b'  # Select one vit model
    rank = 4  # Rank for LoRA adaptation
    module = 'sam_lora_image_encoder'
    class_type = 'LoRA_Sam_v0_v2'
    patch_size=[512, 512]
    z_spacing = 1

    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
   
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # register model
    sam, img_embedding_size = sam_model_registry[vit_name](image_size=img_size,
                                                                    num_classes=num_classes,
                                                                    checkpoint=ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(module)
    
    
    if class_type == "LoRA_Sam_v0_v2":
       net = pkg.LoRA_Sam_v0_v2(sam, rank).cuda()
    else:
       print("wrong class given")

    net.load_lora_parameters(lora_ckpt)

    if num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    
    for filename in os.listdir(input_root):
        if filename.endswith('.mha'):
            output_path = f'{output_root}/{filename}'
            image = image.squeeze(0).cpu().detach().numpy()
            try:
                input_path = input_root + '/' + filename
                image = read(input_path)
                image = np.transpose(image, (2, 0, 1))
                prediction = np.zeros_like(image)
                    
                x, y = image.shape[1], image.shape[2]
                if x != patch_size[0] or y != patch_size[1]:
                    image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
                        
                    inputs = torch.from_numpy(image).unsqueeze(0).float().cuda()
                
                    net.eval()
                    with torch.no_grad():
                        outputs = net(inputs, multimask_output, patch_size[0])
                        output_masks = outputs['masks']
                        
                        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                        
                        out = out.cpu().detach().numpy()
                        
                        out_h, out_w = out.shape
                        
                        
                        if x != out_h or y != out_w:
                            prediction = zoom(out, (x / out_h, y / out_w), order=0)
                            
                        else:
                            prediction = out

                        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
                        prd_itk.SetSpacing((1, 1, z_spacing))
                        sitk.WriteImage(prd_itk, output_path)
            
            except Exception as error:
                        print(error)

if __name__ == '__main__':
    main()
    
