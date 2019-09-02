# In order to get the same results as Detectron's RetinaNet, we want to make sure that given 
# y_hat, y, and locations
# we can compute the same loss value

import unittest
import os
import torch
import numpy as np

def torch_sigmoid_focal_loss(Y_hat, Y, fg_num, gamma=2.0, alpha=0.25, num_classes=80):
    """
    A PyTorch implementation of Sigmoid Focal Loss: 
    Paper: https://arxiv.org/pdf/1708.02002.pdf
    Code:  https://github.com/pytorch/pytorch/blob/master/modules/detectron/sigmoid_focal_loss_op.cu#L31-L66

    Gamma:      2.00 from paper and code
    Alpha:      0.25 from paper and code
    num_clases: 80 classes for COCO
    """
    N = Y_hat.shape[0]
    D = Y_hat.shape[1]
    H = Y_hat.shape[2]
    W = Y_hat.shape[3]
    A = int(D / num_classes)

    # Two weights:
    #   Alpha Weighting for negative and positive examples
    #   Loss weighted according to the total number of positive examples
    zn = (1.0 - alpha) / fg_num
    zp = alpha / fg_num

    expandedTargets = Y.repeat_interleave(80,1)             # Expand Y into the same shape as Y_hat

    aRange = torch.arange(num_classes, dtype=torch.int32)   # Create a range like [0,1,...79]       Shape: (80,)
    repeated = aRange.repeat(A)                             # Tile the range 9 times                Shape: (720,)
    repeated = repeated.view((D,1,1))                       # Reshape so we can broadcast           Shape: (720,1,1)
    zeros = torch.zeros((D, H, W), dtype=torch.int32)       # Create zeros of desired shape         Shape: (720, H, W)
    levelInfo = repeated + zeros                            # Level info represents the class index of the corresponding prediction in Y_hat
    levelInfo = levelInfo.repeat(N,1,1,1)                   # Repeat levelInfo for each image       Shape: (2, 720, H, W)

    # The target classes are in the range 1 - 81 and d is in the range 1-80
    # because we predict A * 80 dim, so for comparison purposes, compare expandedTargets and (levelInfo + 1)
    c1 = expandedTargets  == (levelInfo + 1) 
    c2 = (expandedTargets != -1) & (expandedTargets != (levelInfo + 1))

    #Convert logits to probabilities
    probabilities = 1.0 / (1.0 + torch.exp(-Y_hat))

    # (1 - p) ^ gamma * log(p) where d == (t + 1)
    term1 = torch.pow((1.0 - probabilities), gamma) * torch.log(probabilities)
    # p^gamma * log(1-p)       where d != (t + 1)
    term2 = torch.pow(probabilities, gamma) * torch.log(1 - probabilities)

    loss1 = -(c1.float() * term1 * zp)
    loss2 = -(c2.float() * term2 * zn)

    l1 = torch.sum(loss1)
    l2 = torch.sum(loss2)

    totalLoss = (l1 + l2)
    return totalLoss.numpy()

def numpy_sigmoid_focal_loss(Y_hat, Y, fg_num, gamma=2.0, alpha=0.25, num_classes=80):
    """
    A numpy implementation of Sigmoid Focal Loss: 
    Paper: https://arxiv.org/pdf/1708.02002.pdf
    Code:  https://github.com/pytorch/pytorch/blob/master/modules/detectron/sigmoid_focal_loss_op.cu#L31-L66

    gamma:          2.00 from paper and code
    alpha:          0.25 from paper and code
    num_classes:    80 classes for COCO dataset
    """
    N = Y_hat.shape[0]
    D = Y_hat.shape[1]
    H = Y_hat.shape[2]
    W = Y_hat.shape[3]
    A = int(D / num_classes)

    # Two weights:
    #   Alpha Weighting for negative and positive examples
    #   Loss weighted according to the total number of positive examples
    zn = (1.0 - alpha) / fg_num
    zp = alpha / fg_num

    expandedTargets = np.repeat(Y, num_classes, axis=1)  # Expand Y into the same shape as Y_hat

    aRange = np.arange(num_classes)             # Create a range like [0,1,...79]       Shape: (80,)
    repeated = np.tile(aRange, A)               # Tile the range 9 times                Shape: (720,)
    repeated = repeated.reshape((D,1,1))        # Reshape so we can broadcast           Shape: (720,1,1)
    zeros = np.zeros((D, H, W))                 # Create zeros of desired shape         Shape: (720, H, W)
    levelInfo = repeated + zeros                # Level info represents the class index of the corresponding prediction in Y_hat
    levelInfo = np.tile(levelInfo, (N,1,1,1))   # Repeat levelInfo for each image       Shape: (2, 720, H, W)

    # The target classes are in the range 1 - 81 and d is in the range 1-80
    # because we predict A * 80 dim, so for comparison purposes, compare expandedTargets and (levelInfo + 1)
    c1 = expandedTargets  == (levelInfo + 1) 
    c2 = (expandedTargets != -1) & (expandedTargets != (levelInfo + 1))

    #Convert logits to probabilities
    probabilities = 1.0 / (1.0 + np.exp(-Y_hat))

    # (1 - p) ^ gamma * log(p) where d == (t + 1)
    term1 = np.power((1.0 - probabilities), gamma) * np.log(probabilities)
    # p^gamma * log(1-p)       where d != (t + 1)
    term2 = np.power(probabilities, gamma) * np.log(1 - probabilities)

    loss1 = -(c1 * term1 * zp)
    loss2 = -(c2 * term2 * zn)

    l1 = np.sum(loss1)
    l2 = np.sum(loss2)

    totalLoss = (l1 + l2)
    return totalLoss

def torch_select_smooth_l1_loss(Y_hat, Y, locations, fg_num, beta=0.11):
    """
    A PyTorch port of: https://github.com/pytorch/pytorch/blob/master/modules/detectron/select_smooth_l1_loss_op.cu#L52-L86

    Beta is taken from: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L525
    """

    locations = locations.long()

    y_hat1 = Y_hat[locations[:,0], locations[:,1], locations[:,2], locations[:,3]]
    y_hat2 = Y_hat[locations[:,0], locations[:,1] + 1, locations[:,2], locations[:,3]]
    y_hat3 = Y_hat[locations[:,0], locations[:,1] + 2, locations[:,2], locations[:,3]]
    y_hat4 = Y_hat[locations[:,0], locations[:,1] + 3, locations[:,2], locations[:,3]]

    y_hat = torch.stack([y_hat1, y_hat2, y_hat3,y_hat4], dim=1)

    y1 = Y

    val = y_hat - y1
    abs_val = np.abs(val)

    mask1 = abs_val < beta
    mask2 = ~mask1

    res1 = torch.masked_select(((0.5 * val * val / beta)/fg_num), mask1)
    res2 = torch.masked_select(((abs_val - 0.5 * beta)/fg_num), mask2)

    s1 = res1.sum()
    s2 = res2.sum()
    loss = s1 + s2
    return loss


def naive_select_smooth_l1_loss(Y_hat, Y, locations, fg_num, beta=0.11):
    """
    A Python (CPU) port of: https://github.com/pytorch/pytorch/blob/master/modules/detectron/select_smooth_l1_loss_op.cu#L52-L86

    Beta is taken from: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L525
    """
    M = len(locations)
    L = locations.flatten()

    out = []

    for i in range(M):
        n = int(L[i * 4])
        c = int(L[i * 4 + 1])
        y0 = int(L[i * 4 + 2])
        x = int(L[i * 4 + 3])
        
        for j in range(4):
            y_hat = Y_hat[n,c+j,y0,x]

            y1 = Y[i,j]
            val = y_hat - y1
            abs_val = np.abs(val)
            
            if abs_val < beta:
                out.append((0.5 * val * val / beta)/fg_num)
            else:
                out.append((abs_val - 0.5 * beta)/fg_num)

    return np.sum(out)

def load_select_smooth_l1_test_case_from_file(file):
    test_data = np.load(file, allow_pickle=True)
    y_hat = test_data[0]                        # bbox targets predictions, eg. N x (A * 4) H x W 
    y = test_data[1]                            # true targets: for example: M x 4
    locations = test_data[2]                    # locations of fg boxes: M x 4 (M is # of fg boxes at this level)
    fg_num = test_data[3]                       # Total number of fb boxes across all FPN levels
    detectron_loss = float(test_data[4])        # The loss as calculated by Detectron

    return y_hat, y, locations, fg_num, detectron_loss

def load_focal_loss_test_case_from_file(file):
    test_data = np.load(file, allow_pickle=True)

    y_hat = test_data[0]                       # Class predictions N x (A x C) x H X W
    y = test_data[1]                           # Ground truth      N x A x H x W
    fg_num = test_data[2]                      # Number of positive foreground examples
    detectron_loss = float(test_data[3])       # The loss as calculated by Detectron

    return y_hat, y, fg_num, detectron_loss

class TestStringMethods(unittest.TestCase):

    def test_fpn3_torch_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn3_focal_loss_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        fg_num = torch.tensor(fg_num)
        loss = torch_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_fpn4_torch_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn4_focal_loss_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        fg_num = torch.tensor(fg_num)
        loss = torch_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_fpn5_torch_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn5_focal_loss_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        fg_num = torch.tensor(fg_num)
        loss = torch_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_fpn6_torch_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn6_focal_loss_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        fg_num = torch.tensor(fg_num)
        loss = torch_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_fpn7_torch_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn7_focal_loss_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        fg_num = torch.tensor(fg_num)
        loss = torch_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)

    def test_fpn3_numpy_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn3_focal_loss_test_case.npy")
        loss = numpy_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_fpn4_numpy_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn4_focal_loss_test_case.npy")
        loss = numpy_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_fpn5_numpy_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn5_focal_loss_test_case.npy")
        loss = numpy_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_fpn6_numpy_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn6_focal_loss_test_case.npy")
        loss = numpy_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_fpn7_numpy_focal_loss(self):
        y_hat, y, fg_num, detectron_loss = load_focal_loss_test_case_from_file("fpn7_focal_loss_test_case.npy")
        loss = numpy_sigmoid_focal_loss(y_hat, y, fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=4)
    
    def test_torch_fpn3_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn3_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        locations = torch.tensor(locations)
        loss = torch_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        loss = loss.numpy()
        self.assertAlmostEqual(loss, detectron_loss, places=6)

    def test_torch_fpn4_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn4_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        locations = torch.tensor(locations)
        loss = torch_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        loss = loss.numpy()
        self.assertAlmostEqual(loss, detectron_loss, places=6)

    def test_torch_fpn5_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn5_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        locations = torch.tensor(locations)
        loss = torch_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        loss = loss.numpy()
        self.assertAlmostEqual(loss, detectron_loss, places=6)

    def test_torch_fpn6_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn6_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        locations = torch.tensor(locations)
        loss = torch_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        loss = loss.numpy()
        self.assertAlmostEqual(loss, detectron_loss, places=6)

    def test_torch_fpn7_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn7_test_case.npy")
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)
        locations = torch.tensor(locations)
        loss = torch_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        loss = loss.numpy()
        self.assertAlmostEqual(loss, detectron_loss, places=6)

    def test_fpn3_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn3_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)

    def test_fpn4_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn4_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)
    
    def test_fpn5_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn5_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)
    
    def test_fpn6_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn6_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)
    
    def test_fpn7_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn7_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, fg_num=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)

if __name__ == '__main__':
    unittest.main()