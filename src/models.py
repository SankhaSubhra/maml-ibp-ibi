import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# A regular 4-CONV network
class NetworkModel(nn.Module):

    def __init__(self, k_way):

        # Initialize the network layers.

        super(NetworkModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.batch1 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.batch2 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.batch3 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(64, track_running_stats=False)

        self.lin1 = nn.Linear(64*5*5, k_way)

    def forward(self, x):

        # A forward function only for reference.
        
        x = F.relu(F.max_pool2d(self.batch1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.batch2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.batch3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.batch4(self.conv4(x)), 2))
        
        x = x.view(-1, 64*5*5)
        x = self.lin1(x)
        
        return x

    def functional_forward(self, x, 
        weight_dict, 
        layer_index=None, 
        mixup_flag=None, 
        k_way=None,
        beta_a=None,
        beta_b=None):

        # A functional forward that will actually be used for all requirements. 
        # It only uses functionals thus explicitly needs the weights to be passes. 
        # The functionals can use the regular layer function or their IBP form as required. 

        robust = True
        if layer_index is None:
            y, robust = None, False

        # Block 1
        x = robust_conv_forward(
            x, weight_dict['conv1.weight'], weight_dict['conv1.bias'], stride=1, padding=(1, 1), robust=robust)
        x = robust_batch_norm_forward(
            x, weight_dict['batch1.weight'], weight_dict['batch1.bias'], robust=robust)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        if layer_index == 1:
            y, x, robust = intra_class_mixup(x, mixup_flag, k_way, beta_a, beta_b)

        # Block 2
        x = robust_conv_forward(
            x, weight_dict['conv2.weight'], weight_dict['conv2.bias'], stride=1, padding=(1, 1), robust=robust)
        x = robust_batch_norm_forward(
            x, weight_dict['batch2.weight'], weight_dict['batch2.bias'], robust=robust)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        if layer_index == 2:
            y, x, robust = intra_class_mixup(x, mixup_flag, k_way, beta_a, beta_b)

        # Block 3
        x = robust_conv_forward(
            x, weight_dict['conv3.weight'], weight_dict['conv3.bias'], stride=1, padding=(1, 1), robust=robust)
        x = robust_batch_norm_forward(
            x, weight_dict['batch3.weight'], weight_dict['batch3.bias'], robust=robust)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        if layer_index == 3:
            y, x, robust = intra_class_mixup(x, mixup_flag, k_way, beta_a, beta_b)

        # Block 4
        x = robust_conv_forward(
            x, weight_dict['conv4.weight'], weight_dict['conv4.bias'], stride=1, padding=(1, 1), robust=robust)
        x = robust_batch_norm_forward(
            x, weight_dict['batch4.weight'], weight_dict['batch4.bias'], robust=robust)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        if layer_index == 4:
            y, x, robust = intra_class_mixup(x, mixup_flag, k_way, beta_a, beta_b)

        # Map to number of classes.
        x = x.view(-1, 64*5*5)
        x = F.linear(
            x, weight=weight_dict['lin1.weight'], bias=weight_dict['lin1.bias'])

        return y, x

def robust_conv_forward(x, weight, bias, stride, padding, robust):

    # Convolution function that can propagate interval bounds.

    if robust is False:
        # Regular convolution
        x = F.conv2d(x, weight, bias, stride, padding)
        return x

    # Convolution propagating interval bounds.
    b_size = x.shape[0]//3

    input_p = x[:b_size]
    input_o = x[b_size:2*b_size]
    input_n = x[2*b_size:]

    u = (input_p + input_n)/2
    r = (input_p - input_n)/2

    out_u = F.conv2d(u, weight, bias, stride, padding)
    out_r = F.conv2d(r, torch.abs(weight), None, stride, padding)
    out_o = F.conv2d(input_o, weight, bias, stride, padding)

    return torch.cat([out_u + out_r, out_o, out_u - out_r], 0)

def robust_batch_norm_forward(x, weight, bias, robust):

    # Batch normalization function that can propagate interval bounds.

    if robust is False:
        # Regular batch normalization.
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight, bias=bias, training=True)
        return x

    # Batch normalization propagating interval bounds.
    b_size = x.shape[0]//3
    eps = 1e-5

    input_p = x[:b_size]
    input_o = x[b_size:2*b_size]
    input_n = x[2*b_size:]

    # Equivalent to input_o.mean((0, 2, 3))
    mean = input_o.transpose(0, 1).contiguous().view(
        input_o.shape[1], -1).mean(1)
    var = input_o.transpose(0, 1).contiguous().view(
        input_o.shape[1], -1).var(1, unbiased=False)

    # Element-wise multiplier
    multiplier = torch.rsqrt(var + eps)
    multiplier = multiplier * weight

    offset = (-multiplier * mean) + bias

    multiplier = multiplier.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    offset = offset.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    # Because the scale might be negative, we need to apply a strategy similar to linear
    u = (input_p + input_n)/2
    r = (input_p - input_n)/2

    out_u = torch.mul(u, multiplier) + offset
    out_r = torch.mul(r, torch.abs(multiplier))
    out_o = torch.mul(input_o, multiplier) + offset

    return torch.cat([out_u + out_r, out_o, out_u - out_r], 0)

def intra_class_mixup(y, mixup_flag, k_way, beta_a, beta_b):
    
    # Perform interval bound interpolation
    
    robust = False
    b_size = y.shape[0]//3
    u = y[:b_size]
    l = y[2*b_size:]
    o = y[b_size:2*b_size]

    if mixup_flag is True:

        num_shots = b_size//k_way
        mixup_params = np.repeat(np.random.beta(beta_a, beta_b, k_way), num_shots)
        mixup_params = torch.tensor(mixup_params, dtype=torch.float).view(b_size, 1, 1, 1).to(y.device)

        rand_ext = np.repeat(np.random.randint(0, 2, k_way), num_shots)
        rand_ext = torch.tensor(rand_ext, dtype=torch.float).view(b_size, 1, 1, 1).to(y.device)

        mixup_params_c = 1 - mixup_params
        rand_ext_c = 1 - rand_ext

        ulo = (mixup_params_c*o + 
            mixup_params*rand_ext*u +
            mixup_params*rand_ext_c*l)

        x = torch.cat([o, ulo], 0)
        return y, x, robust
    
    return y, o, robust
