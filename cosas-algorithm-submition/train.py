import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import gc
import torch.backends.cudnn as cudnn
from importlib import import_module
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from trainer import trainer_COSAS
from icecream import ic
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.cuda.amp import GradScaler, autocast

# Set environment variable to configure PyTorch's CUDA memory allocator
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


torch.cuda.empty_cache()
gc.collect()

# Added to reduce memory related issues
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/LarryXu/Synapse/preprocessed_data/train_npz', help='root dir for data')
parser.add_argument('--output', type=str, default='/output/sam/results')
parser.add_argument('--dataset', type=str,
                    default='COSAS', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='/cluster/project7/SAMed/SegCol_UCL/Data/train/train_list.csv', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=200, help='maximum epoch number to train') # switch to 200?
parser.add_argument('--batch_size', type=int,
                    default=6, help='batch_size per gpu') # Reduced batch size from 12 to 6 due to memory issues
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--class_type', type=str, default='LoRA_Sam', help='Type of class')

# Need to specify the task
parser.add_argument('--task', type=int, default= None, help='select 1 or 2')


args = parser.parse_args()

if __name__ == "__main__":
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
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }

    print("Number of classes:", args.num_classes)  # However you've defined this
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

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

    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = 256  # 512 # img_embedding_size * 4

    print(f'This is the resolution of the low res {low_res} and this is the image embedding size{img_embedding_size}')
    
    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    if args.seed != 1234:
        print(f'This is the seed of the traing is {args.seed}')

    with open(config_file, 'w') as f:
        f.writelines(config_items)
    
    # Setting up the logging object
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    if args.task == 1:
        folder_name = ['colorectum', 'pancreas','stomach', 'validation']
    elif args.task == 2:
        folder_name = ['3d-1000', 'kfbio-400', 'teksqray-600p']

    trainer = {'COSAS': trainer_COSAS}
    trainer[dataset_name](args, net, folder_name, snapshot_path, multimask_output, low_res, args.img_size)
    print(f"Snapshot path: {snapshot_path}")