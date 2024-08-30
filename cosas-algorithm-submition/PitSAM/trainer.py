import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from icecream import ic
import gc
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Added to reduce memory related issues
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(True)

# Setting up the logging object
logging = logging.getLogger(__name__)

def print_memory_usage(location):
    '''Implementing a function to monitor memory usage at different points in the code
    Input:
        Location: Write theq location at which the function is called e.g before dataset loading or before backprop
    '''
    print(f"\nMemory usage at {location}:")
    print(torch.cuda.memory_summary(abbreviated=True))
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")

def calc_loss(outputs, label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    
    print(f"The outputs outputs from the model {outputs.keys()}")

    low_res_logits = outputs['low_res_logits']

    # print("Shape of low_res_logits:", low_res_logits.shape)
    
    # Reshape low_res_logits to [B, C, H, W]
    #B, C, H, W = low_res_logits.shape
    #low_res_logits = low_res_logits.view(B, C, H, W)
    #low_res_logits = low_res_logits.view(low_res_logits.shape[0], low_res_logits.shape[1], 256, 256)
    
    # Upsample low_res_logits to match label size
    # Upsample low_res_logits from 128x128 to 256x256
    low_res_logits = F.interpolate(low_res_logits, size=(256, 256), mode='bilinear', align_corners=False)
    
    # Ensure label_batch is the correct shape and type
    label_batch = label_batch.long()
    #print("label_batch shape:", label_batch.shape)
    # Reshape label_batch to match the shape of low_res_logits
    label_batch = label_batch.view(low_res_logits.shape[0], 256, 256)

    # Value to count
    counter_0 = 0
    counter_1 = 1
    counter_255 = 255

    # Method 1: Using np.count_nonzero()
    count_1 = torch.sum(label_batch == counter_1).item()
    

    # Method 2: Using np.sum()
    count_0 = torch.sum(label_batch == counter_0).item()
    count_255 = torch.sum(label_batch == counter_255).item()
    

    #print(f"The number of int 1: {count_1}")
    #print(f"The number of int 0: {count_0}")
    #print(f"The number of int 255: {count_255}")

    #print("low_res_logits shape:", low_res_logits.shape)
    #print("label_batch shape:", label_batch.shape)

    loss_ce = ce_loss(low_res_logits, label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

# Define the forward_with_checkpoint function
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(model, input):
        return checkpoint(model, input)

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: {name} contains NaN or inf values")
        return True
    return False

def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_param_updates(model):
    updates = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            updates.append((name, param.data.clone()))
    return updates

def trainer_COSAS(args, model, folder_name, snapshot_path, multimask_output, low_res, img_size):
    from patch_DataLoader import COSASDataset, RandomGenerator
    gc.enable()
    #logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
    #                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    logging.info(f'This is the number of classes: {num_classes}')
    batch_size = args.batch_size * args.n_gpu
    
    torch.cuda.empty_cache()
    gc.collect()

    # max_iterations = args.max_iterations
    logging.info("Starting dataset loading...")
    
    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),  #Lets try 128*128
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])

    transform_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        # transforms.ToTensor(), Mobarak changed this part
        ])

    # print_memory_usage("Before dataset loading")
    db_train = COSASDataset(args.root_path, folder_name, transform_img=transform_img, transform_mask=transform_mask)
    # print_memory_usage("After dataset loading")

    torch.cuda.empty_cache()
    gc.collect()

    print("The length of train set is: {}".format(len(db_train)))
    logging.info(f"Dataset loaded with {len(db_train)} samples.")
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # print_memory_usage("Before DataLoader creation")
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # print_memory_usage("After DataLoader creation")

    logging.info("Starting model initialization...")
    # print_memory_usage("Before model initialization")
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # print_memory_usage("After model initialization")

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes) #+ 1) removing +1 as there seems to be no background class
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    # logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']  # [b, c, h, w], [b, h, w]
            # print('image_batch, label_batch', image_batch.shape, label_batch.shape)
            # low_res_label_batch = sampled_batch['low_res_label']
        
            # Getting the following error:
            # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same.
            # image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # low_res_label_batch = low_res_label_batch.cuda()

            # Ensure correct dtype and move to GPU
            image_batch = image_batch.float().cuda()
            label_batch = label_batch.float().cuda()
            # low_res_label_batch = low_res_label_batch.float().cuda()

            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            # Error handlind to resolve DoubleTensor error
            if image_batch.dtype != torch.float32:
                logging.error(f"Expected image_batch to be float32, got {image_batch.dtype}")
                raise TypeError(f"Expected image_batch to be float32, got {image_batch.dtype}")
            
            # print_memory_usage(f"Before forward pass, iteration {iter_num}")
            outputs = model(image_batch, multimask_output, args.img_size)
            # print_memory_usage(f"After forward pass, iteration {iter_num}")
            #print(f'This is the label_batch {torch.unique(label_batch)} \n this is the shape {label_batch.shape}')

            loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param) # Replaced: low_res_label_batch
            optimizer.zero_grad()
            
            loss.backward()
            
            # Check gradients for NaN/inf
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if check_nan_inf(param.grad, f"Gradient of {name}"):
                        logging.info(f"NaN/inf detected in gradients at iteration {iteration}. Skipping this batch.")
                        continue
            
            grad_norm = get_gradient_norm(model)
            
            # Check gradient norm for NaN/inf
            #if check_nan_inf(torch.tensor(grad_norm), "Gradient norm"):
            #    logging.info(f"NaN/inf detected in gradient norm at iteration {iteration}. Skipping this batch.")
            #    continue
            
            prev_params = get_param_updates(model)
            
            optimizer.step()
            
            # Log parameter updates
            
            torch.cuda.empty_cache()

            gc.collect()
            
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            # Log the gradient norm
            writer.add_scalar('Gradient/norm', grad_norm, iter_num)

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            if i_batch % 100 == 0:
                logging.info('batch: %d iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, norm: %d' % (i_batch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), grad_norm))
            # logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                
                labs = label_batch[1, ...].unsqueeze(0) 
                labs = labs.unsqueeze(1)
                #print(f'This is the labs variable {labs} and this is labs.shape {labs.shape}')
                #labs = labs.permute(0, 3, 1, 2)  # Reshape to [B, C, H, W]
                labs = labs.float()  # Convert to float tensor
    
                # Remove the extra dimension if num_classes is 1
                if labs.shape[1] == 1:
                    labs = labs.squeeze(1)
    
                # writer.add_image('train/GroundTruth', labs * 50, iter_num)
        logging.info('batch: %d iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (i_batch, iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

    save_interval = 20 # int(max_epoch/6)
    if (epoch_num + 1) % save_interval == 0:
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        try:
            model.save_lora_parameters(save_mode_path)
        except:
            model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"