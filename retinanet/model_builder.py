import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:,:, 0] = torch.clamp(boxes[:,:,0], min=0)
        boxes[:,:, 1] = torch.clamp(boxes[:,:,1], min=0)
        boxes[:,:, 2] = torch.clamp(boxes[:,:,1], min=width)
        boxes[:,:, 3] = torch.clamp(boxes[:,:,1], min=height)

        return boxes

class PyramidFeatures(nn.Module):

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsampled C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class RegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.output(out)

        return out


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))

        out = self.output(out)
        return out

class RetinaNet(nn.Module):

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def __init__(self, num_classes, block, layers):
        super(RetinaNet, self).__init__()

        self.inplanes = 64
        #TODO: Replace with two 3x3 convs
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            x = self.layer2[layers[1]-1].conv3.out_channels
            xx = self.layer3[layers[2]-1].conv3.out_channels,
            xxx = self.layer4[layers[3]-1].conv3.out_channels
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        #self.anchors = Anchors()
        self.anchors = None
        self.regressBoxes = None

        self.clipBoxes = ClipBoxes()
        self.focalLoss = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-np.log((1.0-prior)/prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

    def focal_loss(self, Y_hat, Y, fg_num, gamma=2.0, alpha=0.25, num_classes=80):
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

        cuda = torch.device('cuda')

        expandedTargets = Y.repeat_interleave(80,1)             # Expand Y into the same shape as Y_hat
        expandedTargets = expandedTargets.int()

        aRange = torch.arange(num_classes, dtype=torch.int32)   # Create a range like [0,1,...79]       Shape: (80,)
        aRange = aRange.to(cuda)
        repeated = aRange.repeat(A)                             # Tile the range 9 times                Shape: (720,)
        repeated = repeated.view((D,1,1))                       # Reshape so we can broadcast           Shape: (720,1,1)
        zeros = torch.zeros((D, H, W), dtype=torch.int32)       # Create zeros of desired shape         Shape: (720, H, W)

        zeros = zeros.to(cuda)
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
        return totalLoss

    def select_smooth_l1_loss(self, Y_hat, Y, locations, fg_num, beta=0.11):
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
        abs_val = torch.abs(val)

        mask1 = abs_val < beta
        mask2 = ~mask1

        res1 = torch.masked_select(((0.5 * val * val / beta)/fg_num), mask1)
        res2 = torch.masked_select(((abs_val - 0.5 * beta)/fg_num), mask2)

        s1 = res1.sum()
        s2 = res2.sum()
        loss = s1 + s2
        return loss

    def forward(self, inputs):

        if self.training:
            img_batch, classification_labels, regression_targets, locations, fg_num = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2,x3,x4])

        regression = [self.regressionModel(feature) for feature in features]
        classification = [self.classificationModel(feature) for feature in features]

        totalLoss = 0

        # Select Smooth L1 Regression Loss
        for i in range(len(regression)):
            y_hat = regression[i]
            y = regression_targets[i]
            current_locations = locations[i]
            totalLoss += self.select_smooth_l1_loss(y_hat, y, current_locations, fg_num)

        # Focal Loss
        for i in range(len(classification)):
            y_hat = classification[i]
            y = classification_labels[i]
            totalLoss += self.focal_loss(y_hat, y, fg_num)

        return totalLoss



def create():
    #TODO: Load weights
    #TODO: Don't hardcode classes
    #TODO: Allow other ResNets

    #RetinaNet with ResNet50 backbone
    model = RetinaNet(80, Bottleneck, [3, 4, 6, 3])
    return model