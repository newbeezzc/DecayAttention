import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math


class TemporalDecay(nn.Module):
    def __init__(self, d_in, d_out, diag=False):
        super(TemporalDecay, self).__init__()
        self.flag = 'Conv'
        if self.flag == 'Linear':
            self.diag = diag
            self.W = Parameter(torch.Tensor(d_out, d_in))
            self.b = Parameter(torch.Tensor(d_out))

            if self.diag:
                assert (d_in == d_out)
                m = torch.eye(d_in, d_in)
                self.register_buffer('m', m)

            self.reset_parameters()
        elif self.flag == 'Conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            # self.Conv = nn.Conv1d(in_channels=d_in, out_channels=d_out, kernel_size=3, padding=padding)
            self.Conv = nn.Conv1d(in_channels=d_in, out_channels=d_out, kernel_size=1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.shape[0])
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    @staticmethod
    def compute_delta(delta_fwd, delta_bwd):
        delta_fwd_shifted = torch.zeros_like(delta_fwd).float()
        delta_bwd_shifted = torch.zeros_like(delta_bwd).float()

        delta_fwd_shifted[..., :-1, :] = delta_fwd[..., 1:, :]
        delta_bwd_shifted[..., 1:, :] = delta_bwd[..., :-1, :]

        # alpha = Parameter(torch.Tensor(1))
        # beta = Parameter(torch.Tensor(1))
        alpha = 1.0
        beta = 1.0
        # delta = torch.log(delta_fwd_shifted * delta_bwd_shifted + 1)  # log相乘, 符合正态分布
        # print(delta_fwd_shifted * delta_bwd_shifted)
        # print(delta)
        # exit(0)
        # delta = delta_fwd_shifted * delta_bwd_shifted  # 相乘
        delta = alpha * delta_fwd_shifted + beta * delta_bwd_shifted  # 相加
        return delta

    @staticmethod
    def compute_delta_forward(mask, freq=1):
        delta = torch.zeros_like(mask).float()
        one_step = torch.tensor(freq, dtype=delta.dtype, device=delta.device)
        for i in range(1, delta.shape[-2]):
            m = mask[..., i - 1, :]
            delta[..., i, :] = m * one_step + (1 - m) * torch.add(delta[..., i - 1, :], freq)
        return delta

    @staticmethod
    def compute_delta_backward(mask, freq=1):
        delta = torch.zeros_like(mask).float()
        one_step = torch.tensor(freq, dtype=delta.dtype, device=delta.device)
        for i in range(delta.shape[-2] - 2, -1, -1):
            m = mask[..., i + 1, :]
            delta[..., i, :] = m * one_step + (1 - m) * torch.add(delta[..., i + 1, :], freq)
        return delta

    def forward(self, d):
        if self.flag == 'Linear':
            if self.diag:
                gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
            else:
                gamma = F.relu(F.linear(d, self.W, self.b))
        elif self.flag == "Conv":
            # print(d.permute(0, 2, 1).shape)
            gamma = F.relu(self.Conv(d.permute(0, 2, 1)).transpose(1, 2))
        gamma = torch.exp(-gamma)
        # print(gamma.shape)
        return gamma
