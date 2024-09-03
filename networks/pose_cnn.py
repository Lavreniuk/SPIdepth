from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import timm


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation



class PoseCNN(nn.Module):
    """Relative pose prediction network.
    From SfM-Learner (https://arxiv.org/abs/1704.07813)

    This network predicts the relative pose between two images, concatenated channelwise.
    It consists of a ResNet encoder (with duplicated and scaled input weights) and a simple regression decoder.
    Pose is predicted as axis-angle rotation and a translation vector.

    The objective is to predict the relative pose between two images.
    The network consists of a ResNet encoder (with duplicated weights and scaled for the input images), plus a simple
    regression decoder.
    Pose is predicted as an axis-angle rotation and a translation vector.

    NOTE: Translation is not in metric scale unless training with stereo + mono.

    :param enc_name: (str) `timm` encoder key (check `timm.list_models()`).
    :param pretrained: (bool) If `True`, returns an encoder pretrained on ImageNet.
    """
    def __init__(self, num_input_frames, enc_name: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        #enc_name = 'resnet50' #'convnext_tiny_in22ft1k'
        self.enc_name = enc_name
        self.pretrained = pretrained

        self.n_imgs = num_input_frames
        self.encoder = timm.create_model(enc_name, in_chans=3 * self.n_imgs, features_only=True, pretrained=pretrained)
        self.n_chenc = self.encoder.feature_info.channels()

        self.squeeze = self.block(self.n_chenc[-1], 256, kernel_size=1)
        self.decoder = nn.Sequential(
            self.block(256, 256, kernel_size=3, stride=1, padding=1),
            self.block(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 6 * (self.n_imgs-1), kernel_size=1),
        )

    @staticmethod
    def block(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Module:
        """Conv + ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        """Pose network forward pass.

        :param x: (Tensor) (b, 2*3, h, w) Channel-wise concatenated input images.
        :return: (dict[str, Tensor]) {
            R: (b, 2, 3) Predicted rotation in axis-angle (direction=axis, magnitude=angle).
            t: (b, 2, 3) Predicted translation.
        }
        """
        feat = self.encoder(x)
        out = self.decoder(self.squeeze(feat[-1]))
        out = 0.01 * out.mean(dim=(2, 3)).view(-1, self.n_imgs-1, 1, 6)
        return out[..., :3], out[..., 3:]
