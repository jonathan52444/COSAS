import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry

from patch_DataLoader import COSASDataset, RandomGenerator
from torchvision import transforms
# from icecream import ic

def inference(multimask_output, db_config, net, test_save_path=None):
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        ])

    #transform_mask = transforms.Compose([
        #transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
        # transforms.ToTensor(), Mobarak changed this part
    #    ])
    
    db_test = db_config['Dataset'](volume_path, folder_name, transform_img=transform_img, inference=True)
    
    if len(db_test) == 0:
        logging.error(f"No data found in the provided folders: {folder_name}. Please check the dataset paths.")
        return 0
    
    print("The length of train set is: {}".format(len(db_test)))
    logging.info(f"Dataset loaded with {len(db_test)} samples.")

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    net.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name, original_size  = sampled_batch['image'], sampled_batch['mask'], sampled_batch['case_name'], sampled_batch['original_size']
        metric_i = test_single_volume(image, label, net, classes=num_classes, multimask_output=multimask_output,
                                      patch_size=[img_size, img_size], input_size=[input_size, input_size],
                                      original_size = original_size, test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'])
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    # Define default variables
    config = None  # The config file provided by the trained model
    volume_path = '/cluster/project7/SAMed/samed_codes/SAMed/testset/test_vol_h5/'
    dataset = 'COSAS'  # Experiment name
    num_classes = 2
    list_dir = '/cluster/project7/SAMed/samed_codes/SAMed/lists/lists_Synapse/'
    output_dir = '/output'
    img_size = 512  # Input image size of the network
    input_size = 256  # The input size for training SAM model
    seed = 1234  # Random seed
    is_savenii = False  # Whether to save results during inference
    deterministic = 1  # Whether to use deterministic training
    ckpt = '/cluster/project7/SAMed/samed_codes/SAMed/checkpoints/sam_vit_b_01ec64.pth'  # Pretrained checkpoint
    lora_ckpt = '/cluster/project7/SAMed/samed_codes/SAMed/checkpoints/epoch_159.pth'  # The checkpoint from LoRA
    vit_name = 'vit_b'  # Select one vit model
    rank = 4  # Rank for LoRA adaptation
    module = 'sam_lora_image_encoder'

    # You can now use these variables in your code as needed

    class_type = 'LoRA_Sam_v0_v2'

    # Need to specify the task
    # parser.add_argument('--task', type=int, default= None, help='select 1 or 2 or 3 for both')
    

    # if config is not None:
    #     # overwtite default configurations with config file\
    #     config_dict = config_to_dict(config)
    #     for key in config_dict:
    #         setattr(args, key, config_dict[key])

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
    dataset_name = dataset
    dataset_config = {
        'COSAS': {
            'Dataset': COSASDataset,
            'volume_path': volume_path,
            'list_dir': list_dir,
            'num_classes': num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # initialize log
    log_folder = os.path.join(output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    #logging.info(f'\nTesting for model state at {test_epoch}\n')
    # logging.info(str(args))

    if is_savenii:
        test_save_path = os.path.join(output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    
    inference(multimask_output, dataset_config[dataset_name], net, test_save_path)
