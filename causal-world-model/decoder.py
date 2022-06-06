import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits
from torch.distributions import Normal, kl_divergence
from utils import linear_annealing, spatial_transform, calc_kl_z_pres_bernoulli
from vision_modules import NumericalRelaxedBernoulli
from common import  *

class Decoder(nn.Module):

    def __init__(self, glimpse_dec_net):
        super(Decoder, self).__init__()
        self.glimpse_dec = glimpse_dec_net

    def forward(self, bg, z_all, eps=1e-15):
        z_what, z_where, z_depth, z_pres, z_coordinate = unwrap_z(z_all)
        bs = z_what.size(0)
        max_num_obj = z_what.size(1)
        bns = bs * max_num_obj

        z_what = z_what.reshape(bns, -1)
        z_where = z_where.reshape(bns, -1)
        z_depth = z_depth.reshape(bns, -1)
        z_pres = z_pres.reshape(bns, -1)

        # (bs * num_cell_h * num_cell_w, dim, glimpse_size, glimpse_size)
        o_att, alpha_att = self.glimpse_dec(z_what)
        alpha_att_hat = alpha_att * z_pres.reshape(-1, 1, 1, 1)
        y_att = alpha_att_hat * o_att

        # (bns, 3, img_h, img_w)
        y_each_obj = spatial_transform(y_att, z_where, (bns, 3, img_h, img_w), inverse=True)
        y_each_obj = y_each_obj.reshape(bs, -1, 3, img_h, img_w)
        # (batch_size_t, 1, glimpse_size, glimpse_size)
        importance_map_low_res = alpha_att_hat * torch.sigmoid(-z_depth).reshape(-1, 1, 1, 1)
        # (batch_size_t, 1, img_h, img_w)
        importance_map = spatial_transform(importance_map_low_res, z_where, (bns, 1, img_h, img_w),
                                                    inverse=True)
        importance_map = importance_map.reshape(bs, -1, 1, img_h, img_w)
        importance_map_norm = importance_map / (importance_map.sum(dim=1, keepdim=True) + eps)

        # (batch_size_t, 1, img_h, img_w)
        alpha_map = spatial_transform(alpha_att_hat, z_where, (bns, 1, img_h, img_w), inverse=True)
        alpha_map = alpha_map.reshape(bs, -1, 1, img_h, img_w)
        #final_z_pres_mask = z_pres.squeeze() * obj_mask.view(bns)

        # (bs, 1, img_h, img_w)
        alpha_map_sum = alpha_map.sum(dim=1)
        alpha_map_sum = alpha_map_sum + (alpha_map_sum.clamp(eps, 1 - eps) - alpha_map_sum).detach()
        y_nobg = (y_each_obj * importance_map_norm).sum(dim=1)
        y = y_nobg + (1 - alpha_map_sum) * bg
        return y_each_obj, y, alpha_map, alpha_map_sum, importance_map, importance_map_norm