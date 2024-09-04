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


class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'}


def inference(args, multimask_output, db_config, net, folder_name, test_save_path=None):
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        ])

    #transform_mask = transforms.Compose([
        #transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
        # transforms.ToTensor(), Mobarak changed this part
    #    ])
    
    db_test = db_config['Dataset'](args.volume_path, folder_name, transform_img=transform_img, inference=True)
    
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
        metric_i = test_single_volume(image, label, net, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      original_size = original_size, test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'])
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes + 1):
        try:
            logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))
        except:
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info("Testing Finished!")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='/cluster/project7/SAMed/samed_codes/SAMed/testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='COSAS', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--list_dir', type=str, default='/cluster/project7/SAMed/samed_codes/SAMed/lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=256, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/cluster/project7/SAMed/samed_codes/SAMed/checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='/cluster/project7/SAMed/samed_codes/SAMed/checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    # parser.add_argument('--test_epoch', type=str, default = 'epoch_199')
    parser.add_argument('--class_type', type=str, default='LoRA_Sam_v0_v2', help='Type of class')

    # Need to specify the task
    parser.add_argument('--task', type=int, default= None, help='select 1 or 2 or 3 for both')
    
    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'COSAS': {
            'Dataset': COSASDataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    
    
    if args.class_type == "LoRA_Sam":
       net = pkg.LoRA_Sam(sam, args.rank).cuda()
    elif args.class_type == "LoRA_Sam_v0_v2":
       net = pkg.LoRA_Sam_v0_v2(sam, args.rank).cuda()
    else:
       print("wrong class given")

    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    #logging.info(f'\nTesting for model state at {args.test_epoch}\n')
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    if args.task == 1:
        folder_name = ['colorectum', 'pancreas','stomach', 'validation']
    elif args.task == 2:
        folder_name = ['3d-1000', 'kfbio-400', 'teksqray-600p']
    elif args.task == 3:
        folder_name = ['colorectum', 'pancreas','stomach', 'validation', '3d-1000', 'kfbio-400', 'teksqray-600p']
    elif args.task == 4:
        folder_name = ['validation']
    
    inference(args, multimask_output, dataset_config[dataset_name], net, folder_name, test_save_path)
