import os
import math
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PropNet(nn.Module):

    def __init__(self, node_dim_in, edge_dim_in, nf_hidden, node_dim_out, edge_dim_out,
                 edge_type_num=1, pstep=2, batch_norm=1, use_gpu=True):

        super(PropNet, self).__init__()

        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.nf_hidden = nf_hidden

        self.node_dim_out = node_dim_out
        self.edge_dim_out = edge_dim_out

        self.edge_type_num = edge_type_num
        self.pstep = pstep

        # node encoder
        modules = [
            nn.Linear(node_dim_in, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_encoder = nn.Sequential(*modules)

        # edge encoder
        self.edge_encoders = nn.ModuleList()
        for i in range(edge_type_num):
            modules = [
                nn.Linear(node_dim_in * 2 + edge_dim_in, nf_hidden),
                nn.ReLU()]
            if batch_norm == 1:
                modules.append(nn.BatchNorm1d(nf_hidden))

            self.edge_encoders.append(nn.Sequential(*modules))

        # node propagator
        modules = [
            # input: node_enc, node_rep, edge_agg
            nn.Linear(nf_hidden * 3, nf_hidden),
            nn.ReLU(),
            nn.Linear(nf_hidden, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_propagator = nn.Sequential(*modules)

        # edge propagator
        self.edge_propagators = nn.ModuleList()
        for i in range(pstep):
            edge_propagator = nn.ModuleList()
            for j in range(edge_type_num):
                modules = [
                    # input: node_rep * 2, edge_enc, edge_rep
                    nn.Linear(nf_hidden * 3, nf_hidden),
                    nn.ReLU(),
                    nn.Linear(nf_hidden, nf_hidden),
                    nn.ReLU()]
                if batch_norm == 1:
                    modules.append(nn.BatchNorm1d(nf_hidden))
                edge_propagator.append(nn.Sequential(*modules))

            self.edge_propagators.append(edge_propagator)

        # node predictor
        modules = [
            nn.Linear(nf_hidden * 2, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, node_dim_out))
        self.node_predictor = nn.Sequential(*modules)

        # edge predictor
        modules = [
            nn.Linear(nf_hidden * 2, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, edge_dim_out))
        self.edge_predictor = nn.Sequential(*modules)

    def forward(self, node_rep, edge_rep=None, edge_type=None, start_idx=0,
                ignore_node=False, ignore_edge=False):
        # node_rep: B x N x node_dim_in
        # edge_rep: B x N x N x edge_dim_in
        # edge_type: B x N x N x edge_type_num
        # start_idx: whether to ignore the first edge type
        B, N, _ = node_rep.size()

        # node_enc
        node_enc = self.node_encoder(node_rep.view(-1, self.node_dim_in)).view(B, N, self.nf_hidden)

        # edge_enc
        node_rep_r = node_rep[:, :, None, :].repeat(1, 1, N, 1)
        node_rep_s = node_rep[:, None, :, :].repeat(1, N, 1, 1)
        if edge_rep is not None:
            tmp = torch.cat([node_rep_r, node_rep_s, edge_rep], 3)
        else:
            tmp = torch.cat([node_rep_r, node_rep_s], 3)

        edge_encs = []
        for i in range(start_idx, self.edge_type_num):
            edge_enc = self.edge_encoders[i](tmp.view(B * N * N, -1)).view(B, N, N, 1, self.nf_hidden)
            edge_encs.append(edge_enc)
        # edge_enc: B x N x N x edge_type_num x nf
        edge_enc = torch.cat(edge_encs, 3)

        if edge_type is not None:
            edge_enc = edge_enc * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]

        # edge_enc: B x N x N x nf
        edge_enc = edge_enc.sum(3)

        node_effect = node_enc
        edge_effect = edge_enc

        for i in range(self.pstep):
            if i == 0:
                node_effect = node_enc
                edge_effect = edge_enc

            # calculate edge_effect
            node_effect_r = node_effect[:, :, None, :].repeat(1, 1, N, 1)
            node_effect_s = node_effect[:, None, :, :].repeat(1, N, 1, 1)
            tmp = torch.cat([node_effect_r, node_effect_s, edge_effect], 3)

            edge_effects = []
            for j in range(start_idx, self.edge_type_num):
                edge_effect = self.edge_propagators[i][j](tmp.view(B * N * N, -1))
                edge_effect = edge_effect.view(B, N, N, 1, self.nf_hidden)
                edge_effects.append(edge_effect)
            # edge_effect: B x N x N x edge_type_num x nf
            edge_effect = torch.cat(edge_effects, 3)

            if edge_type is not None:
                edge_effect = edge_effect * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]

            # edge_effect: B x N x N x nf
            edge_effect = edge_effect.sum(3)

            # calculate node_effect
            edge_effect_agg = edge_effect.sum(2)
            tmp = torch.cat([node_enc, node_effect, edge_effect_agg], 2)
            node_effect = self.node_propagator(tmp.view(B * N, -1)).view(B, N, self.nf_hidden)

        node_effect = torch.cat([node_effect, node_enc], 2).view(B * N, -1)
        edge_effect = torch.cat([edge_effect, edge_enc], 3).view(B * N * N, -1)

        # node_pred: B x N x node_dim_out
        # edge_pred: B x N x N x edge_dim_out
        if ignore_node:
            edge_pred = self.edge_predictor(edge_effect)
            return edge_pred.view(B, N, N, -1)
        if ignore_edge:
            node_pred = self.node_predictor(node_effect)
            return node_pred.view(B, N, -1)

        node_pred = self.node_predictor(node_effect).view(B, N, -1)
        edge_pred = self.edge_predictor(edge_effect).view(B, N, N, -1)
        return node_pred, edge_pred

