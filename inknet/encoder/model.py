from torch import nn
from collections import OrderedDict


class InkEncoder(nn.Module):
    def __init__(self):
        super(InkEncoder, self).__init__()

        self.feature_space_size = 2048

        # NOTE: attribute names and OrderedDict keys below must not be changed —
        # they are baked into the state_dict of saved checkpoints.
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', make_conv_block(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', make_conv_block(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', make_conv_block(256, 384, 3, pad=1)),
            ('conv4', make_conv_block(384, 384, 3, pad=1)),
            ('conv5', make_conv_block(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', make_dense_block(256 * 3 * 5, 2048)),
            ('fc2', make_dense_block(self.feature_space_size, self.feature_space_size)),
        ]))

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.shape[0], 256 * 3 * 5)
        out = self.fc_layers(out)
        return out


class InkEncoderThin(nn.Module):
    def __init__(self):
        super(InkEncoderThin, self).__init__()

        self.feature_space_size = 1024

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', make_conv_block(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', make_conv_block(96, 128, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', make_conv_block(128, 128, 3, pad=1)),
            ('conv4', make_conv_block(128, 128, 3, pad=1)),
            ('conv5', make_conv_block(128, 128, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', make_dense_block(128 * 3 * 5, self.feature_space_size)),
            ('fc2', make_dense_block(self.feature_space_size, self.feature_space_size)),
        ]))

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.shape[0], 128 * 3 * 5)
        out = self.fc_layers(out)
        return out


class InkEncoderCompact(nn.Module):
    def __init__(self):
        super(InkEncoderCompact, self).__init__()

        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', make_conv_block(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', make_conv_block(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', make_conv_block(256, 384, 3, pad=1)),
            ('conv5', make_conv_block(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', make_dense_block(256 * 3 * 5, 2048)),
        ]))

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.shape[0], 256 * 3 * 5)
        out = self.fc_layers(out)
        return out


def make_conv_block(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))


def make_dense_block(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))
