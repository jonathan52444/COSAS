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
from segment_anything.build_sam import sam_model_registry

from torchvision import transforms
import SimpleITK as sitk
import SimpleITK
from scipy.ndimage import zoom
from PIL import Image
# from icecream import ic
import logging
logging.basicConfig(level=logging.INFO)

def read(path):
    image = SimpleITK.ReadImage(path)
    return SimpleITK.GetArrayFromImage(image)


def write(path, array):
    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(image, path, useCompression=False)

import sys
import os

def print_os_paths():
    logging.info("Current working directory:")
    logging.info(os.getcwd())
    logging.info("\nPython executable path:")
    logging.info(sys.executable)
    logging.info("\nAll system paths:")
    for index, path in enumerate(sys.path, start=1):
        logging.info(f"{index}. {path}")

def print_directory_contents():
    current_dir = os.getcwd()
    logging.info(f"Contents of directory: {current_dir}")
    logging.info("-" * 50)

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path):
            logging.info(f"[DIR]  {item}")
        elif os.path.isfile(item_path):
            logging.info(f"[FILE] {item}")
        else:
            logging.info(f"[OTHER] {item}")

def main():
    # Define default variables
    input_root = '/Users/sirbucks/Desktop/Coding/Workspaces/COSAS/cosas-algorithm-submition/input/images/adenocarcinoma-image'
    output_root = '/Users/sirbucks/Desktop/Coding/Workspaces/COSAS/cosas-algorithm-submition/output/images/adenocarcinoma-mask'
    #cosas-algorithm-submition/output/images/adenocarcinoma-mask
    
    num_classes = 2
    img_size = 512  # Input image size of the network
    input_size = 256  # The input size for training SAM model
    seed = 1234  # Random seed
    deterministic = 1  # Whether to use deterministic training
    ckpt = '/Users/sirbucks/Desktop/Coding/Workspaces/COSAS/cosas-algorithm-submition/sam_vit_b_01ec64.pth' # 'sam_vit_b_01ec64.pth'  # Pretrained checkpoint
    lora_ckpt = '/Users/sirbucks/Desktop/Coding/Workspaces/COSAS/cosas-algorithm-submition/checkpoint_199_512.pth' # 'checkpoint_199_512.pth'  # The checkpoint from LoRA
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
   
    # if not os.path.exists(output_root):
    #     logging.info('The path does not exist')
    #     os.makedirs(output_root)

    # register model
    sam, img_embedding_size = sam_model_registry[vit_name](image_size=img_size,
                                                                    num_classes=num_classes,
                                                                    checkpoint=ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(module)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if class_type == "LoRA_Sam_v0_v2":
        net = pkg.LoRA_Sam_v0_v2(sam, rank).to(device)
    else:
        logging.info("wrong class given")
    
    logging.info('after loading the parameters')
    net.load_lora_parameters(lora_ckpt)
    logging.info('after loading the parameters')
    if num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    transform_img = transforms.Compose([
        transforms.Resize((512, 512)),  #Lets try 128*128
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])
    logging.info('Entering the filename loop')
    for filename in os.listdir(input_root):
        if filename.endswith('.mha'):
            output_path = f'{output_root}/{filename}'
            try:
                input_path = input_root + '/' + filename
                image = read(input_path)

                prediction = np.zeros_like(image)
                x, y = image.shape[0], image.shape[1]
                logging.info(f'This is the image shape {image.shape}')

                # Convert to PIL image
                image = Image.fromarray(image)
                
                transform_img = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                        )
                    ])
                image = transform_img(image)
                
                inputs = image.unsqueeze(0).float().to(device)
                # if x != patch_size[0] or y != patch_size[1]:
                #     image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
                        
                #inputs = torch.from_numpy(image).unsqueeze(0).float().cuda()
                
                net.eval()
                with torch.no_grad():
                    outputs = net(inputs, multimask_output, patch_size[0])
                    output_masks = outputs['masks']
                        
                    out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                        
                    out = out.cpu().detach().numpy()
                    out_h, out_w = out.shape
                    logging.info(f'This is the out shape {out.shape}')
                        
                    if x != out_h or y != out_w:
                        prediction = zoom(out, (x / out_h, y / out_w), order=0)
                        logging.info(f'This is the out shape {prediction.shape}')    
                    else:
                        prediction = out
                        #logging.info(f'This is the out shape {prediction.shape}') 
                    
                    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
                    prd_itk.SetSpacing((1, 1, z_spacing))
                    sitk.WriteImage(prd_itk, output_path)
            
            except Exception as error:
                        logging.info(error)

if __name__ == '__main__':
    main()
    
