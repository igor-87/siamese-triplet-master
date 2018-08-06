import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import math
import random
import warnings

import torch
def xavier_uniform_(tensor, gain=1):
    r"""Fills the input `Tensor` with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where
    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}
    Also known as Glorot initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-a, a)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_normal_(tensor, gain=1):
    r"""Fills the input `Tensor` with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where
    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}
    Also known as Glorot initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0, std)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
class InceptionBased(nn.Module):
    def __init__(self, feature_size=2048, im_size=299, normalize=False):
        super(InceptionBased, self).__init__()
        self.normalize = normalize
        self.im_size = 299
        self.feature_size = feature_size
        self.inception = torchvision.models.inception_v3(pretrained=True)
        self.inception.fc = nn.Linear(2048, feature_size)
        xavier_normal_(self.inception.fc.weight)

    def forward(self, x):
        # y = self.inception(x)
        ## weird result in training mode, probably a bug in inception module?
        # if self.training:
        #    if self.normalize:
        #        return y[0]/torch.norm(y[0],2,1).repeat(1, self.feature_size)
        #    else:
        #        return y[0]
        # else:
        #    if self.normalize:
        #        return y/torch.norm(y,2,1).repeat(1, self.feature_size)
        #    else:
        #        return y
        if self.inception.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.inception.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.inception.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.inception.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.inception.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.inception.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.inception.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.inception.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.inception.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.inception.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6e(x)
        # 17 x 17 x 768
        if self.inception.training and self.inception.aux_logits:
            aux = self.inception.AuxLogits(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.inception.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.inception.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # x = x.view(-1, self.feature_size)

        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.inception.fc(x)
        if self.normalize:
            return x / torch.norm(x, 2, 1).repeat(1, self.feature_size)
        else:
            return x

    def SetLearningRate(self, lr1, lr2):
        print('Setting learning rate for inception net')
        d = [
            {'params': self.inception.Conv2d_1a_3x3.parameters(), 'lr': lr1},
            {'params': self.inception.Conv2d_2a_3x3.parameters(), 'lr': lr1},
            {'params': self.inception.Conv2d_2b_3x3.parameters(), 'lr': lr1},
            {'params': self.inception.Conv2d_3b_1x1.parameters(), 'lr': lr1},
            {'params': self.inception.Conv2d_4a_3x3.parameters(), 'lr': lr1},
            {'params': self.inception.Mixed_5b.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_5c.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_5d.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_6a.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_6b.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_6c.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_6d.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_6e.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.AuxLogits.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_7a.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_7b.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.Mixed_7c.parameters(), 'lr': 2 * lr1},
            {'params': self.inception.fc.parameters(), 'lr': lr2},
        ]
        return d
