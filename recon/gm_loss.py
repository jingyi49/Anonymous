import torch.nn as nn
import torch


class GM(nn.Module):
    """
     Simplified version of the VGG19 "feature" block
     This module's only job is to return the "feature loss" for the inputs
    """

    def __init__(self, channel_in=3, width=64):
        super(GM, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)

        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)

        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.load_params_()
        
    def gm(self, x):
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        G = torch.bmm(features, features.transpose(1, 2)) 
        return G / (C * H * W)
    
    def load_params_(self):
        # Download and load Pytorch's pre-trained weights
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        for ((name, source_param), target_param) in zip(state_dict.items(), self.parameters()):
            target_param.data = source_param.data
            target_param.requires_grad = False

    def feature_loss(self, x):
        target = x[:x.shape[0] // 2]
        pred = x[x.shape[0] // 2:]
        target_gm = self.gm(target)
        pred_gm = self.gm(pred)
        return (target_gm - pred_gm).pow(2).mean()

    def forward(self, x):
        """
        :param x: Expects x to be the target and source to concatenated on dimension 0
        :return: Feature loss
        """
        feature_weights = {
            'conv1': 1.0,
            'conv2': 0.9,
            'conv3': 0.8,
            'conv4': 0.7,
            'conv5': 0.6,
            'conv6': 0.5,
            'conv7': 0.4,
            'conv8': 0.3,
            'conv9': 0.2,
            'conv10': 0.1,
            'conv11': 0.05,
            'conv12': 0.05,
            'conv13': 0.01,
            'conv14': 0.01,
            'conv15': 0.01,
            'conv16': 0.01
        }
        
        x = self.conv1(x)
        loss = feature_weights['conv1'] * self.feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += feature_weights['conv2'] *self.feature_loss(x)
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        loss += feature_weights['conv3'] *self.feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += feature_weights['conv4'] *self.feature_loss(x)
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        loss += feature_weights['conv5'] *self.feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += feature_weights['conv6'] *self.feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += feature_weights['conv7'] * self.feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += feature_weights['conv8'] *self.feature_loss(x)
        x = self.mp(self.relu(x))  # 16x16

        x = self.conv9(x)
        loss += feature_weights['conv9'] *self.feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += feature_weights['conv10'] *self.feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += feature_weights['conv11'] *self.feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += feature_weights['conv12'] * self.feature_loss(x)
        x = self.mp(self.relu(x))  # 8x8

        x = self.conv13(x)
        loss += feature_weights['conv13'] * self.feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += feature_weights['conv14'] *self.feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += feature_weights['conv15'] *self.feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += feature_weights['conv16'] *self.feature_loss(x)

        return loss