"""
A UNet architecture that largely mimics the original network published in 2015.

Binary segmentation (foreground and background) only. That is, only a single object from a single class will be
segmented.

This version has extra layers, in the form of batch normalisation, applied.

At its widest there are 1024 channels.

The final layer of this network does NOT make use of the sigmoid function. This was causing a lot of trouble with
the nn.BCELoss() function which cannot handle the limits of 0 and 1 due to the log nature of the loss calculation.
Instead, a different loss function must be used (nn.BCEWithLogitsLoss() is an example of many), that requires
the output of the network having NOT gone through a sigmoid layer as the loss function will do that itself.
"""

import torch as t
import torch.nn.functional as F
from torch.nn import ConvTranspose2d, BatchNorm2d, Conv2d, Module, MaxPool2d
from torch.nn.init import kaiming_normal_


class UNet(Module):
    def __init__(self):
        """Initialise the model layers."""
        super(UNet, self).__init__()
        self.cv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.cv2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.cv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2d(128)
        self.cv4 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = BatchNorm2d(128)
        self.cv5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = BatchNorm2d(256)
        self.cv6 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = BatchNorm2d(256)
        self.cv7 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn7 = BatchNorm2d(512)
        self.cv8 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn8 = BatchNorm2d(512)
        self.cv9 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn9 = BatchNorm2d(1024)
        self.cv10 = Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.bn10 = BatchNorm2d(1024)

        self.cv_t1 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.cv11 = Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.bn11 = BatchNorm2d(512)
        self.cv12 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn12 = BatchNorm2d(512)
        self.cv_t2 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.cv13 = Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn13 = BatchNorm2d(256)
        self.cv14 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn14 = BatchNorm2d(256)
        self.cv_t3 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.cv15 = Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn15 = BatchNorm2d(128)
        self.cv16 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn16 = BatchNorm2d(128)
        self.cv_t4 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.cv17 = Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn17 = BatchNorm2d(64)
        self.cv18 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn18 = BatchNorm2d(64)
        self.cv_final = Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))

        self.max_pool = MaxPool2d(kernel_size=2)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise the weights of the filters randomly."""
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """Forward pass of the network."""
        # First layer (contraction).
        c1 = F.relu(self.bn1(self.cv1(x)))
        c1 = F.relu(self.bn2(self.cv2(c1)))
        p1 = self.max_pool(c1)
        # Second layer (contraction).
        c2 = F.relu(self.bn3(self.cv3(p1)))
        c2 = F.relu(self.bn4(self.cv4(c2)))
        p2 = self.max_pool(c2)
        # Third layer (contraction).
        c3 = F.relu(self.bn5(self.cv5(p2)))
        c3 = F.relu(self.bn6(self.cv6(c3)))
        p3 = self.max_pool(c3)
        # Fourth layer (contraction).
        c4 = F.relu(self.bn7(self.cv7(p3)))
        c4 = F.relu(self.bn8(self.cv8(c4)))
        p4 = self.max_pool(c4)
        # Fifth layer (contraction).
        c5 = F.relu(self.bn9(self.cv9(p4)))
        c5 = F.relu(self.bn10(self.cv10(c5)))
        # Sixth layer (expansion).
        u6 = self.cv_t1(c5)
        u6 = F.interpolate(u6, size=c4.shape[2:])
        u6 = t.cat([u6, c4], dim=1)
        c6 = F.relu(self.bn11(self.cv11(u6)))
        c6 = F.relu(self.bn12(self.cv12(c6)))
        # Seventh layer (expansion).
        u7 = self.cv_t2(c6)
        u7 = F.interpolate(u7, size=c3.shape[2:])
        u7 = t.cat([u7, c3], dim=1)
        c7 = F.relu(self.bn13(self.cv13(u7)))
        c7 = F.relu(self.bn14(self.cv14(c7)))
        # Eighth layer (expansion).
        u8 = self.cv_t3(c7)
        u8 = F.interpolate(u8, size=c2.shape[2:])
        u8 = t.cat([u8, c2], dim=1)
        c8 = F.relu(self.bn15(self.cv15(u8)))
        c8 = F.relu(self.bn16(self.cv16(c8)))
        # Ninth layer (expansion).
        u9 = self.cv_t4(c8)
        u9 = F.interpolate(u9, size=c1.shape[2:])
        u9 = t.cat([u9, c1], dim=1)
        c9 = F.relu(self.bn17(self.cv17(u9)))
        c9 = F.relu(self.bn18(self.cv18(c9)))
        # Output layer.
        # outputs = t.sigmoid(self.cv_final(c9))
        outputs = self.cv_final(c9)
        return outputs
