"""
This code is refer from:
https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/transformation.py
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class TPS(nn.Module):
    def __init__(
        self, 
        in_channels, 
        num_fiducial, 
        model_name
    ):
        super(TPS, self).__init__()
        self.loc_net = LocalizationNetwork(
            in_channels, num_fiducial, model_name)
        self.grid_generator = GridGenerator(
            self.loc_net.out_channels, num_fiducial)
        self.out_channels = in_channels
    
    def forward(self, x):
        batch_C_prime = self.loc_net(x)  # batch_size x num_fiducial x 2
        batch_P_prime = self.grid_generator(
            batch_C_prime, x.shape[2:])  # batch_size x (width x height) x 2
        batch_P_prime = batch_P_prime.reshape(
            -1, x.shape[2], x.shape[3], 2)
        batch_I_r = F.grid_sample(
            x, batch_P_prime, padding_mode="border")
        return batch_I_r


class LocalizationNetwork(nn.Module):
    def __init__(
        self, 
        in_channels, 
        num_fiducial, 
        model_name
    ):
        super(LocalizationNetwork, self).__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == "large":
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64
        
        self.block_list = nn.ModuleList()
        for idx, num_filters in enumerate(num_filters_list):
            self.block_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, num_filters, 3, 1, 1, bias=False), 
                    nn.BatchNorm2d(num_filters), 
                    nn.ReLU(inplace=True)
                    )
                )
            if idx == len(num_filters_list) - 1:
                pool = nn.AdaptiveAvgPool2d(1)
            else:
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.block_list.append(pool)
            in_channels = num_filters
        self.loc_fc1 = nn.Sequential(
            nn.Linear(in_channels, fc_dim), 
            nn.ReLU(inplace=True)
        )
        self.loc_fc2 = nn.Linear(fc_dim, F * 2)
        self.out_channels = F * 2

        # Init fc2 in LocalizationNetwork
        self.loc_fc2.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.loc_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        for block in self.block_list:
            x = block(x)
        x = x.reshape(batch_size, -1)
        x = self.loc_fc2(self.loc_fc1(x))
        x = x.reshape(batch_size, self.F, 2)
        return x


class GridGenerator(nn.Module):
    def __init__(self, in_channels, num_fiducial):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.F = num_fiducial
        self.fc = nn.Linear(in_channels, 6)

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self._build_C()
        P = self._build_P(I_r_size)
        ## for multi-gpu, you need register buffer
        self.register_buffer(
            "inv_delta_C", 
            torch.tensor(
                self._build_inv_delta_C(C)).float())  # F+3 x F+3
        self.register_buffer(
            "P_hat", 
            torch.tensor(
                self._build_P_hat(C, P)).float())  # n x F+3
        ## for fine-tuning with different image width, you may use below instead of self.register_buffer
        #self.inv_delta_C = torch.tensor(self._build_inv_delta_C(self.F, self.C)).float().cuda()  # F+3 x F+3
        #self.P_hat = torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float().cuda()  # n x F+3

        batch_size = batch_C_prime.shape[0]
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)

        batch_C_ex_part = self.get_expand(batch_C_prime)

        batch_C_prime_with_zeros = torch.cat((
            batch_C_prime, 
            batch_C_ex_part.float()
            ), dim=1)  # batch_size x F+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C.to(batch_C_prime_with_zeros.device), 
            batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(
            batch_P_hat.to(batch_T.device), 
            batch_T)  # batch_size x n x 2
        
        return batch_P_prime  # batch_size x n x 2

    
    def _build_C(self):
        """ Return coordinates of fiducial points in I_r; C """
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C

    def _build_P(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        I_r_grid_x = (np.arange(
            -I_r_width, I_r_width, 2) + 1.0) / I_r_width     # self.I_r_width
        I_r_grid_y = (np.arange(
            -I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(
            np.meshgrid(I_r_grid_x, I_r_grid_y), 
            axis=2)    # self.I_r_width x self.I_r_height x 2
        return P.reshape([-1, 2])  # (I_r_width x I_r_height) x 2

    def _build_inv_delta_C(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # F+3 x F+3
        delta_C = np.concatenate([
            np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
            np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
            np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ], 
            axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P_hat(self, C, P):
        F = self.F
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def get_expand(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape(B, H * C)
        batch_C_ex_part = self.fc(batch_C_prime)
        batch_C_ex_part = batch_C_ex_part.reshape(-1, 3, 2)
        return batch_C_ex_part
