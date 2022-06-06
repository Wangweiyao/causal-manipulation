import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits
from torch.distributions import Normal, kl_divergence
from utils import linear_annealing, spatial_transform, calc_kl_z_pres_bernoulli
from vision_modules import NumericalRelaxedBernoulli
from common import *


class ProposalCore(nn.Module):

    def __init__(self, args):
        super(ProposalCore, self).__init__()

        self.args = args
        self.z_pres_bias = 0
        if img_w == 64:
            if self.args.num_cell_h == 8:
                self.mask_enc_net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, nf, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, nf)
                )
            elif self.args.num_cell_h == 4:
                self.mask_enc_net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, nf, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, nf)
                )
        elif img_w == 128:
            if self.args.num_cell_h == 8:
                self.mask_enc_net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, nf, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, nf)
                )
            elif self.args.num_cell_h == 4:
                self.mask_enc_net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, nf, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, nf)
                )

        self.img_mask_cat_enc = nn.Sequential(
            nn.Conv2d(img_encode_dim + nf, img_encode_dim, 1),
            nn.CELU(),
            nn.GroupNorm(16, img_encode_dim),
            nn.Conv2d(img_encode_dim, img_encode_dim, 1),
            nn.CELU(),
            nn.GroupNorm(16, img_encode_dim),
        )
        if img_w == 64:
            self.z_where_net = nn.Conv2d(img_encode_dim, (z_where_shift_dim + z_where_scale_dim) * 2, 1)

            self.z_pres_net = nn.Conv2d(img_encode_dim, z_pres_dim, 1)

            self.z_depth_net = nn.Conv2d(img_encode_dim, z_depth_dim * 2, 1)

        elif img_w == 128:
            self.z_where_net = nn.Sequential(
                nn.Conv2d(img_encode_dim, 64, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, (z_where_shift_dim + z_where_scale_dim) * 2, 1)
            )

            self.z_pres_net = nn.Sequential(
                nn.Conv2d(img_encode_dim, 64, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, z_pres_dim, 1)
            )

            self.z_depth_net = nn.Sequential(
                nn.Conv2d(img_encode_dim, 32, 1),
                nn.CELU(),
                nn.GroupNorm(8, 32),
                nn.Conv2d(32, z_depth_dim * 2, 1)
            )
        if self.args.num_cell_h == 8:
            offset_y, offset_x = torch.meshgrid([torch.arange(8.), torch.arange(8.)])
        elif self.args.num_cell_h == 4:
            offset_y, offset_x = torch.meshgrid([torch.arange(4.), torch.arange(4.)])

        self.register_buffer('offset', torch.stack((offset_x, offset_y), dim=0))

    def forward(self, img_enc, alpha, tau, gen_pres_probs=None, gen_depth_mean=None,
                gen_depth_std=None, gen_where_mean=None, gen_where_std=None):
        """

        :param x: (bs, dim, img_h, img_w)
        :param propagate_encode: (bs, nf)
        :param tau:
        :return:
        """

        bs = img_enc.size(0)

        mask_enc = self.mask_enc_net(alpha)

        x_alpha_enc = torch.cat((img_enc, mask_enc), dim=1)

        cat_enc = self.img_mask_cat_enc(x_alpha_enc)

        # (bs, 1, 8, 8)
        z_pres_logits = pres_logit_factor * torch.tanh(self.z_pres_net(cat_enc) + self.z_pres_bias)

        # (bs, dim, 8, 8)
        z_depth_mean, z_depth_std = self.z_depth_net(cat_enc).chunk(2, 1)
        z_depth_std = F.softplus(z_depth_std)
        # (bs, 4 + 4, 8, 8)
        z_where_mean, z_where_std = self.z_where_net(cat_enc).chunk(2, 1)
        z_where_std = F.softplus(z_where_std)

        q_z_pres = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=tau)
        z_pres_y = q_z_pres.rsample()

        z_pres = torch.sigmoid(z_pres_y)

        q_z_depth = Normal(z_depth_mean, z_depth_std)

        z_depth = q_z_depth.rsample()

        q_z_where = Normal(z_where_mean, z_where_std)

        z_where = q_z_where.rsample()

        # (bs, dim, 8, 8)
        z_where_origin = z_where.clone()

        scale, ratio = z_where[:, :2].tanh().chunk(2, 1)
        scale = self.args.size_anc + self.args.var_s * scale
        ratio = self.args.ratio_anc + self.args.var_anc * ratio
        ratio_sqrt = ratio.sqrt()
        z_where[:, 0:1] = scale / ratio_sqrt
        z_where[:, 1:2] = scale * ratio_sqrt
        z_where[:, 2:] = 2. / self.args.num_cell_h * (self.offset + 0.5 + z_where[:, 2:].tanh()) - 1

        z_where = z_where.permute(0, 2, 3, 1).reshape(-1, 4)

        return z_where, z_pres, z_depth, z_where_mean, z_where_std, \
               z_depth_mean, z_depth_std, z_pres_logits, z_pres_y, z_where_origin


class ProposalRejectionCell(nn.Module):

    def __init__(self, args, img_encoder, z_what_net, glimpse_dec_net):
        super(ProposalRejectionCell, self).__init__()
        self.args = args

        self.z_pres_anneal_start_step = 0000
        self.z_pres_anneal_end_step = 500
        self.z_pres_anneal_start_value = 1e-1
        self.z_pres_anneal_end_value = self.args.z_pres_anneal_end_value
        self.z_pres_masked_prior = 1e-8
        self.max_num_obj = args.max_num_obj

        self.ProposalNet = ProposalCore(self.args)
        self.img_encoder = img_encoder
        self.z_what_net = z_what_net
        self.glimpse_dec = glimpse_dec_net
        self.z_coordinate_net = nn.Sequential(
            nn.Linear(z_all_dim-z_coordinate_dim, nf),
            nn.CELU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, z_coordinate_dim),
        )
        self.register_buffer('prior_what_mean', torch.zeros(1))
        self.register_buffer('prior_what_std', torch.ones(1))
        self.register_buffer('prior_bg_mean', torch.zeros(1))
        self.register_buffer('prior_bg_std', torch.ones(1))
        self.register_buffer('prior_depth_mean', torch.zeros(1))
        self.register_buffer('prior_depth_std', torch.ones(1))
        self.register_buffer('prior_where_mean',
                             torch.tensor([.1, .1, 0., 0.]).view((z_where_scale_dim + z_where_shift_dim), 1, 1))
        self.register_buffer('prior_where_std',
                             torch.tensor([1., 1., 1., 1.]).view((z_where_scale_dim + z_where_shift_dim), 1, 1))
        self.register_buffer('prior_z_pres_prob', torch.tensor(self.z_pres_anneal_start_value))
        self.register_buffer('num_cell', torch.tensor(self.args.num_cell_h * self.args.num_cell_w))

    @property
    def p_z_what(self):
        return Normal(self.prior_what_mean, self.prior_what_std)

    @property
    def p_z_depth(self):
        return Normal(self.prior_depth_mean, self.prior_depth_std)

    @property
    def p_z_where(self):
        return Normal(self.prior_where_mean, self.prior_where_std)

    def forward(self, x, eps=1e-15):
        """
            :param x: (bs, 3, img_h, img_w)
        """
        bs = x.size(0)
        device = x.device
        ids_prop = torch.zeros(bs, 2).to(device)
        lengths = torch.zeros(bs).to(device)
        alpha_map = torch.zeros(bs, 2, 1, img_h, img_w).to(device)
        
        alpha_map_sum = alpha_map.sum(1)
        alpha_map_sum = \
            alpha_map_sum + (alpha_map_sum.clamp(eps, 1 - eps) - alpha_map_sum).detach()
        
        max_num_disc_obj = (self.max_num_obj - lengths).long()

        self.prior_z_pres_prob = linear_annealing(self.args.global_step, self.z_pres_anneal_start_step,
                                                  self.z_pres_anneal_end_step, self.z_pres_anneal_start_value,
                                                  self.z_pres_anneal_end_value, device)

        img_enc = self.img_encoder(x)
        # z_where: (bs * num_cell_h * num_cell_w, 4)
        # z_pres, z_depth, z_pres_logits: (bs, dim, num_cell_h, num_cell_w)
        z_where, z_pres, z_depth, z_where_mean, z_where_std, \
        z_depth_mean, z_depth_std, z_pres_logits, z_pres_y, z_where_origin = self.ProposalNet(
            img_enc, alpha_map_sum, self.args.tau, gen_pres_probs=x.new_ones(1) * self.args.gen_disc_pres_probs,
            gen_depth_mean=self.prior_depth_mean, gen_depth_std=self.prior_depth_std,
            gen_where_mean=self.prior_where_mean, gen_where_std=self.prior_where_std
        )
        num_cell_h, num_cell_w = z_pres.shape[2], z_pres.shape[3]

        q_z_where = Normal(z_where_mean, z_where_std)
        q_z_depth = Normal(z_depth_mean, z_depth_std)

        z_pres_orgin = z_pres

        # (bs * num_cell_h * num_cell_w, 3, glimpse_size, glimpse_size)
        x_att = spatial_transform(torch.stack(num_cell_h * num_cell_w * (x,), dim=1).view(-1, 3, img_h, img_w),
                                    z_where,
                                    (bs * num_cell_h * num_cell_w, 3, glimpse_size, glimpse_size), inverse=False)

        # (bs * num_cell_h * num_cell_w, dim)
        z_what_mean, z_what_std = self.z_what_net(x_att)
        z_what_std = F.softplus(z_what_std)

        q_z_what = Normal(z_what_mean, z_what_std)

        z_what = q_z_what.rsample()

        # The following "if" is useful only if you don't have high-memery GPUs, better to remove it if you do
        if phase_obj_num_contrain:
            z_pres = z_pres.view(bs, -1)

            z_pres_threshold = z_pres.sort(dim=1, descending=True)[0][torch.arange(bs), max_num_disc_obj]

            z_pres_mask = (z_pres > z_pres_threshold.view(bs, -1)).float()

            z_pres = z_pres * z_pres_mask

            z_pres = z_pres.view(bs, 1, num_cell_h, num_cell_w)

        # (bs * num_cell_h * num_cell_w, dim, glimpse_size, glimpse_size)
        o_att, alpha_att = self.glimpse_dec(z_what)

        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)

        y_att = alpha_att_hat * o_att

        # (bs * num_cell_h * num_cell_w, 3, img_h, img_w)
        y_each_cell = spatial_transform(y_att, z_where, (bs * num_cell_h * num_cell_w, 3, img_h, img_w),
                                    inverse=True)

        # (bs * num_cell_h * num_cell_w, 1, glimpse_size, glimpse_size)
        importance_map = alpha_att_hat * torch.sigmoid(-z_depth).view(-1, 1, 1, 1)
        # importance_map = -z_depth.view(-1, 1, 1, 1).expand_as(alpha_att_hat)
        # (bs * num_cell_h * num_cell_w, 1, img_h, img_w)
        importance_map_full_res = spatial_transform(importance_map, z_where,
                                                    (bs * num_cell_h * num_cell_w, 1, img_h, img_w),
                                                    inverse=True)

        # (bs * num_cell_h * num_cell_w, 1, img_h, img_w)
        alpha_map = spatial_transform(alpha_att_hat, z_where, (bs * num_cell_h * num_cell_w, 1, img_h, img_w),
                                      inverse=True)

        # (bs * num_cell_h * num_cell_w, z_what_dim)
        kl_z_what = kl_divergence(q_z_what, self.p_z_what) * z_pres_orgin.view(-1, 1)
        # (bs, num_cell_h * num_cell_w, z_what_dim)
        kl_z_what = kl_z_what.view(-1, num_cell_h * num_cell_w, z_what_dim)
        # (bs * num_cell_h * num_cell_w, z_depth_dim)
        kl_z_depth = kl_divergence(q_z_depth, self.p_z_depth) * z_pres_orgin
        # (bs, num_cell_h * num_cell_w, z_depth_dim)
        kl_z_depth = kl_z_depth.view(-1, num_cell_h * num_cell_w, z_depth_dim)
        # (bs, dim, num_cell_h, num_cell_w)
        kl_z_where = (kl_divergence(q_z_where, self.p_z_where) * z_pres_orgin)#.mean(axis=[2,3])

        kl_z_pres = calc_kl_z_pres_bernoulli(z_pres_logits, self.prior_z_pres_prob)

        kl_z_pres = kl_z_pres.view(-1, num_cell_h * num_cell_w, z_pres_dim)

        # (bs, num_cell_h * num_cell_w)
        ids = torch.arange(num_cell_h * num_cell_w).view(1, -1).expand(bs, -1).to(x.device).float() + \
              ids_prop.max(dim=1, keepdim=True)[0] + 1

        if self.args.log_phase:
            self.log = {
                'z_what': z_what,
                'z_where': z_where,
                'z_pres': z_pres,
                'z_pres_logits': z_pres_logits,
                'z_what_std': q_z_what.stddev,
                'z_what_mean': q_z_what.mean,
                'z_where_std': q_z_where.stddev,
                'z_where_mean': q_z_where.mean,
                'x_att': x_att,
                'y_att': y_att,
                'prior_z_pres_prob': self.prior_z_pres_prob.unsqueeze(0),
                'o_att': o_att,
                'alpha_att_hat': alpha_att_hat,
                'alpha_att': alpha_att,
                'y_each_cell': y_each_cell,
                'z_depth': z_depth,
                'z_depth_std': q_z_depth.stddev,
                'z_depth_mean': q_z_depth.mean,
                # 'importance_map_full_res_norm': importance_map_full_res_norm,
                'z_pres_y': z_pres_y,
                'ids': ids
            }
        else:
            self.log = {}

        z_coordinate_input = torch.cat([z_what, z_where, z_depth.view(-1,z_depth_dim), z_pres.view(-1,z_pres_dim)], dim=-1)
        z_coordinate = self.z_coordinate_net(z_coordinate_input)
        z_all = torch.cat([z_coordinate_input, z_coordinate], dim=-1).view(bs, num_cell_h * num_cell_w, -1)
        '''
        z_what, z_where, z_depth, z_pres = unwrap_z(z_all)
        z_mask = (z_pres > 0.7).float()
        num_obj_each = torch.sum(z_mask, dim=1)
        max_num_obj = int(torch.max(num_obj_each).item())
        max_num_obj = max(max_num_obj, 2)

        z_all_pre = torch.zeros(bs, max_num_obj, z_all_dim).to(device)
        z_mask_pre = torch.zeros(bs, max_num_obj, z_pres_dim).to(device)
        ids_pre = torch.zeros(bs, max_num_obj).to(device)

        for b in range(bs):
            num_obj = int(num_obj_each[b])
            idx = z_mask[b].nonzero()[:, 0]
            z_all_pre[b, :num_obj] = z_all[b, idx]
            z_mask_pre[b, :num_obj] = z_mask[b, idx]
            ids_pre[b, :num_obj] = ids[b, idx]
        lengths = torch.sum(z_mask_pre, dim=(1, 2)).reshape(bs)
        '''
        return z_all, ids, \
               kl_z_what.flatten(start_dim=1).sum(dim=1), \
               kl_z_where.flatten(start_dim=1).sum(dim=1), \
               kl_z_pres.flatten(start_dim=1).sum(dim=1), \
               kl_z_depth.flatten(start_dim=1).sum(dim=1), \
               self.log