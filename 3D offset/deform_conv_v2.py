import torch
from torch import nn
import numpy as np
from torch.autograd import Variable, Function

import glob
import os
import scipy.io as io
import argparse
import sys

class DeformConv3d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):

        super(DeformConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ConstantPad3d(padding, 0)
        N = kernel_size ** 3
        self.conv = nn.Conv3d(inc * N, outc, kernel_size=1, bias=bias)             
        self.p_conv = nn.Conv3d(inc, 3*N, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv3d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)        
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 3

        if self.padding:
            x = self.zero_padding(x)   
        p = self._get_p(offset, dtype)       
        p = p[:, :, ::self.stride, ::self.stride, ::self.stride]
        p = p.contiguous().permute(0, 2, 3, 4, 1) 
       
        q_sss = Variable(p.data, requires_grad=False).floor()  

        q_lll = q_sss + 1 
        q_sss = torch.cat([
            torch.clamp(q_sss[..., :N], 0, x.size(2) - 1), 
            torch.clamp(q_sss[..., N:2 * N], 0, x.size(3) - 1),  
            torch.clamp(q_sss[..., 2 * N:], 0, x.size(4) - 1)  
        ], dim=-1).long()
        q_lll = torch.cat([
            torch.clamp(q_lll[..., :N], 0, x.size(2) - 1), 
            torch.clamp(q_lll[..., N:2 * N], 0, x.size(3) - 1),  
            torch.clamp(q_lll[..., 2 * N:], 0, x.size(4) - 1)  
        ], dim=-1).long()
        q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)


        mask = torch.cat([
            p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
            p[..., N:2 * N].lt(self.padding) + p[..., N:2 * N].gt(x.size(3) - 1 - self.padding),
            p[..., 2 * N:].lt(self.padding) + p[..., 2 * N:].gt(x.size(4) - 1 - self.padding),
        ], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))  
        p = p * (1 - mask) + floor_p * mask

        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),
            torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1),
        ], dim=-1)

        g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        x_q_sss = self._get_x_q(x, q_sss, N)
        x_q_lll = self._get_x_q(x, q_lll, N)
        x_q_ssl = self._get_x_q(x, q_ssl, N)
        x_q_sls = self._get_x_q(x, q_sls, N)
        x_q_sll = self._get_x_q(x, q_sll, N)
        x_q_lss = self._get_x_q(x, q_lss, N)
        x_q_lsl = self._get_x_q(x, q_lsl, N)
        x_q_lls = self._get_x_q(x, q_lls, N)

        x_offset = g_sss.unsqueeze(dim=1) * x_q_sss + \
                   g_lll.unsqueeze(dim=1) * x_q_lll + \
                   g_ssl.unsqueeze(dim=1) * x_q_ssl + \
                   g_sls.unsqueeze(dim=1) * x_q_sls + \
                   g_sll.unsqueeze(dim=1) * x_q_sll + \
                   g_lss.unsqueeze(dim=1) * x_q_lss + \
                   g_lsl.unsqueeze(dim=1) * x_q_lsl + \
                   g_lls.unsqueeze(dim=1) * x_q_lls
        


        x_offset = self._reshape_x_offset(x_offset, ks)               
        out = self.conv(x_offset)
        return out,p

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing='ij')

        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)
       
        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype): 
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)

        return p_0

    def _get_p(self, offset, dtype): 
        N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        p_n = self._get_p_n(N, dtype).to(offset)
        p_0 = self._get_p_0(h, w, d, N, dtype).to(offset)
        p = p_0 + p_n + offset

        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()

        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
 
        x = x.contiguous().view(b, c, -1)      
        index = q[..., :N] * padded_w * padded_d + q[..., N:2 * N] * padded_d + q[..., 2 * N:]

        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, d, N = x_offset.size()
        x_offset = x_offset.permute(0, 1, 5, 2, 3, 4)
        x_offset = x_offset.contiguous().view(b, c * N, h, w, d)
        return x_offset            
