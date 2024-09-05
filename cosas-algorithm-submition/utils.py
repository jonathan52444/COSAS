import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
import SimpleITK as sitk


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, multimask_output, patch_size=[512, 512], input_size=[224, 224],
                       original_size = None, test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        print(f'Before reshaping the array this is the image type {type(image)} and this is the type of image.shape {type(image.shape)}')
        print('The code went to the if statement in the test_single_volume function')
        image = np.transpose(image, (2, 0, 1))
        prediction = np.zeros_like(label)
        
        print(f'This is the image shape {image.shape} and this is the image type {type(image)} and this is the type of image.shape {type(image.shape)}')
        print(f'This is the case_name {case}')
        x, y = image.shape[1], image.shape[2]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
            
        inputs = torch.from_numpy(image).unsqueeze(0).float().cuda()
        print(f'This is the shape of the inputs to the neural net {inputs.shape}')
        # inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            print(f'Shape of the output masks: {output_masks.shape}')
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            print(f'Shape of after argmax is applied: {out.shape}')
            out = out.cpu().detach().numpy()
            print(f'Shape after out is reformated: {out.shape} and this is the type {type(out)}')
            out_h, out_w = out.shape
            print(f'this is the output height {out_h} and this is the output width {out_w}')
            #if (out_h, out_w) != (original_size[0], original_size[1]):
            #    prediction = zoom(out, (original_size[0] / prediction.shape[0],
            #                    original_size[1] / prediction.shape[1]), order=0)
            if x != out_h or y != out_w:
                prediction = zoom(out, (x / out_h, y / out_w), order=0)
                print(f"This is the prediction after update's shape {prediction.shape}")
            else:
                prediction = out
                print(f"This is the prediction after update's shape {prediction.shape}")
    else:
        print('The code went to the else statement in the test_single_volume function')
        print(f'This is the image shape {image.shape}')
        x, y = image.shape[-2:]
        prediction = np.zeros_like(label)
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y, 1), order=3)
        inputs = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if prediction.shape != (original_size[1], original_size[0]):
                prediction = zoom(prediction, (original_size[1] / prediction.shape[0],
                                        original_size[0] / prediction.shape[1]), order=0)

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))

        print(f"type of test_save_path: {type(test_save_path)}, value: {test_save_path}")
        print(f"type of case: {type(case)}, value: {case}")
        print(f"type of prd_itk: {type(prd_itk)}") #, value: {prd_itk}")

        sitk.WriteImage(prd_itk, test_save_path + '/' + case[0])
    return 1