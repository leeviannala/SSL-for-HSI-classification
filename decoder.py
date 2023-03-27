from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import Tensor
from torch.nn import _reduction as _Reduction
from torch.overrides import has_torch_function_variadic, handle_torch_function
import warnings
from torchmetrics.functional.regression.mape import _mean_absolute_percentage_error_compute
from torchmetrics.functional.regression.mape import _mean_absolute_percentage_error_update

class DualNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward():
        pass


def IoU(inputt, target, smooth=1):
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        inputt = inputt.view(-1)
        target = target.view(-1)
        intersection = (inputt * target).sum()
        total = (inputt + target).sum()
        union = total - intersection 
        
        ret = (intersection + smooth)/(union + smooth)
        return ret

class DualLoss(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(DualLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input_hsi: Tensor, target_hsi: Tensor, 
                input_class: Tensor, target_class: Tensor, 
                smooth: float =1.0, epsilon: float =1.17e-06) -> Tensor:
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(
            input_class, target_class, epsilon=epsilon)
        mape = _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)
        iou = 1 - IoU(input_class, target_class, smooth=smooth)
        return mape + iou
    
    
class Conv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    
class ConvDecoder(nn.Module):
    '''
    Input is a*b*c, output is a*b*d where c << d. So we need to deconvolve the spectral dimension. 
    a*b*c -> 3a * 3b * 3c -> ... -> 3a * 3b * d -> average pool 3,3 -> a*b*d 
    '''
    def conv_block(self, in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, activation=nn.LeakyReLU):
        return nn.Sequential(
            nn.Conv3D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm3d(in_channels=out_channels),
            activation()
        )
    def deconv_block(in_channels    = 1, 
                     out_channels   = 1, 
                     kernel_size    = 1, 
                     stride         = 1, 
                     padding        = 0, 
                     dilation       = 1, 
                     output_padding = 0,
                     activation=nn.LeakyReLU):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels    = in_channels, 
                               out_channels   = out_channels, 
                               kernel_size    = kernel_size, 
                               stride         = stride, 
                               padding        = padding, 
                               dilation       = dilation, 
                               output_padding = output_padding),
            nn.BatchNorm3d(in_channels=out_channels),
            activation()
    )
    def deconv_2D_block(in_channels    = 1, 
                     out_channels   = 1, 
                     kernel_size    = 1, 
                     stride         = 1, 
                     padding        = 0, 
                     dilation       = 1, 
                     output_padding = 0,
                     activation=nn.LeakyReLU):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels    = in_channels, 
                               out_channels   = out_channels, 
                               kernel_size    = kernel_size, 
                               stride         = stride, 
                               padding        = padding, 
                               dilation       = dilation, 
                               output_padding = output_padding),
            nn.BatchNorm3d(in_channels=out_channels),
            activation()
    )
    def __init__(self, parameters) -> None:
        super().__init__()
        self.conv1 = self.conv_block(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.deconv1 = self.deconv_block(in_channels=32, out_channels=32, kernel_size=(3, 1, 1), stride=(3,1,1), padding=0, dilation=1, output_padding=0)
        self.conv2 = self.conv_block(in_channels=32, out_channels=16, kernel_size=(1, 5, 5), stride=1, padding=(0,2,2), dilation=1)
        self.conv3 = self.conv_block(in_channels=16, out_channels=8, kernel_size=(5,1,1), padding=(2,0,0), stride=1, dilation=1)
        self.conv4 = self.conv_block(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.deconv2 = self.deconv_2D_block(in_channels = 3*parameters['input_bands'], 
                                            out_channels=parameters['output_bands'], 
                                            kernel_size=1, padding=0, output_padding=0, dilation=1, stride=1, activation=nn.Sigmoid)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.deconv2(x)
        return x
        
