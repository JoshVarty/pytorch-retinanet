# In order to get the same results as Detectron's RetinaNet, we want to make sure that given 
# y_hat, y, and locations
# we can compute the same loss value

import unittest
import numpy as np
import os

def numpy_sigmoid_focal_loss(Y_hat, Y, weight, gamma, alpha):
    """
    A numpy implementation of Sigmoid Focal Loss: 
    Paper: https://arxiv.org/pdf/1708.02002.pdf
    Code:  https://github.com/pytorch/pytorch/blob/master/modules/detectron/sigmoid_focal_loss_op.cu#L31-L66
    """
    N = Y_hat.shape[0]
    D = Y_hat.shape[1]
    H = Y_hat.shape[2]
    W = Y_hat.shape[3]

    num_classes = 80
    A = int(D / num_classes)

    # Two weights:
    #   Alpha Weighting for negative and positive examples
    #   Loss weighted according to the total number of positive examples
    zn = (1.0 - alpha) / weight
    zp = alpha / weight

    expandedTargets = np.repeat(Y, num_classes, 1)  # Expand Y into the same shape as Y_hat

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

def naive_select_smooth_l1_loss(Y_hat, Y, locations, S, beta=0.11):
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
                out.append((0.5 * val * val / beta)/S)
            else:
                out.append((abs_val - 0.5 * beta)/S)

    return np.sum(out)

def load_select_smooth_l1_test_case_from_file(file):
    test_data = np.load(file, allow_pickle=True)
    y_hat = test_data[0]                        # bbox targets predictions, eg. N x (A * 4) H x W 
    y = test_data[1]                            # true targets: for example: M x 4
    locations = test_data[2]                    # locations of fg boxes: M x 4 (M is # of fg boxes at this level)
    fg_num = test_data[3]                       # Total number of fb boxes across all FPN levels
    detectron_loss = test_data[4]               # The loss as calculated by Detectron

    return y_hat, y, locations, fg_num, detectron_loss

class TestStringMethods(unittest.TestCase):

    def test_fpn3_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn3_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, S=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)

    def test_fpn4_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn4_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, S=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)
    
    def test_fpn5_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn5_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, S=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)
    
    def test_fpn6_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn6_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, S=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)
    
    def test_fpn7_bbox_select_smooth_l1_loss(self):
        y_hat, y, locations, fg_num, detectron_loss = load_select_smooth_l1_test_case_from_file("select_smooth_l1_fpn7_test_case.npy")
        loss = naive_select_smooth_l1_loss(y_hat, y, locations, S=fg_num)
        self.assertAlmostEqual(loss, detectron_loss, places=6)

if __name__ == '__main__':
    unittest.main()