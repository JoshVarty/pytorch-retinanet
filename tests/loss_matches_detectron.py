# In order to get the same results as Detectron's RetinaNet, we want to make sure that given 
# y_hat, y, and locations
# we can compute the same loss value

import unittest
import numpy as np
import os


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

def load_test_case_from_file(file):
    test_data = np.load(file, allow_pickle=True)
    y_hat = test_data[0]                        # bbox targets predictions, eg. N x (A * 4) H x W 
    y = test_data[1]                            # true targets: for example: M x 4
    locations = test_data[2]                    # locations of fg boxes: M x 4 (M is # of fg boxes at this level)
    fg_num = test_data[3]                       # Total number of fb boxes across all FPN levels
    detectron_loss = test_data[4]               # The loss as calculated by Detectron

    return y_hat, y, locations, fg_num, detectron_loss

class TestStringMethods(unittest.TestCase):

    def test_fpn3_bbox_select_l1_loss(self):

        y_hat, y, locations, fg_num, detectron_loss = load_test_case_from_file("fpn3_test_case.npy")

        loss = naive_select_smooth_l1_loss(y_hat, y, locations, S=fg_num)

        self.assertAlmostEqual(loss, detectron_loss, places=6)

if __name__ == '__main__':
    unittest.main()