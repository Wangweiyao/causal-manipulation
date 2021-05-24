import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from discovery import ProposalRejectionCell
from vision_modules import ImgEncoder, ZWhatEnc, GlimpseDec, BgDecoder, BgEncoder, ConvLSTMEncoder
from common import *
import random
from propagation import PropagationCell  

class SCALOR(nn.Module):

    def __init__(self, args):
        super(SCALOR, self).__init__()
        self.args = args
        self.bg_what_std_bias = 0

        self.image_enc = ImgEncoder(args)
        self.z_what_net = ZWhatEnc()
        self.glimpse_dec_net = GlimpseDec()      

        self.propagate_cell = PropagationCell(
                args,
                z_what_net=self.z_what_net,
                glimpse_dec_net=self.glimpse_dec_net
        )
        if not self.args.phase_no_background:
            self.bg_enc = BgEncoder()
            self.bg_dec = BgDecoder()

        self.proposal_rejection_cell = ProposalRejectionCell(
                args,
                z_what_net=self.z_what_net,
                glimpse_dec_net=self.glimpse_dec_net
        )
        '''
        if args.phase_parallel:
            self.image_enc = nn.DataParallel(self.image_enc)
            self.propagate_cell = nn.DataParallel(self.propagate_cell)
            self.bg_enc = nn.DataParallel(self.bg_enc)
            self.bg_dec = nn.DataParallel(self.bg_dec)
            self.proposal_rejection_cell = nn.DataParallel(self.proposal_rejection_cell)
        '''
        self.register_buffer('z_pres_threshold', torch.tensor(.7))
        self.register_buffer('prior_bg_mean_t1', torch.zeros(1))
        self.register_buffer('prior_bg_std_t1', torch.ones(1))
        self.register_buffer('color_t', self.args.color_t)

    @property
    def p_bg_what_t1(self):
        return Normal(self.prior_bg_mean_t1, self.prior_bg_std_t1)

    def forward(self, seq, actions, eps=1e-15):
        bs = seq.size(0)
        seq_len = seq.size(1)
        device = seq.device
        self.device = seq.device

        z_what_pre = seq.new_zeros(bs, 1, z_what_dim)
        z_where_pre = seq.new_zeros(bs, 1, (z_where_scale_dim + z_where_scale_dim))
        z_where_bias_pre = seq.new_zeros(bs, 1, (z_where_scale_dim + z_where_scale_dim))
        z_depth_pre = seq.new_zeros(bs, 1, z_depth_dim)
        z_pres_pre = seq.new_zeros(bs, 1, z_pres_dim)
        cumsum_one_minus_z_pres_pre = seq.new_zeros(bs, 1, z_pres_dim)
        ids_pre = seq.new_zeros(bs, 1)

        lengths = seq.new_zeros(bs)

        kl_z_pres_all = seq.new_zeros(bs, seq_len)
        kl_z_what_all = seq.new_zeros(bs, seq_len)
        kl_z_where_all = seq.new_zeros(bs, seq_len)
        kl_z_depth_all = seq.new_zeros(bs, seq_len)
        kl_z_bg = seq.new_zeros(bs, seq_len)
        kl_edge_type_all = seq.new_zeros(bs, seq_len)
        log_imp_all = seq.new_zeros(bs, seq_len)
        log_like_all = seq.new_zeros(bs, seq_len)
        y_seq = seq.new_zeros(bs, seq_len, 3, img_h, img_w)

        log_list = []
        log_list = []
        scalor_log_list = []
        counting_list = []

        img_enc_seq = self.image_enc(seq.reshape(-1, seq.size(-3), seq.size(-2), seq.size(-1)))
        img_enc_seq = img_enc_seq.reshape(bs, -1, img_enc_seq.size(-3), img_enc_seq.size(-2), img_enc_seq.size(-1))
        self.propagate_cell.node_type = None # wait to initilize inside propagation

        # Background
        z_bg_mean = torch.zeros([bs,bg_what_dim]).to(device)
        z_bg_std = torch.ones([bs,bg_what_dim]).to(device)
        z_bg_std = F.softplus(z_bg_std + self.bg_what_std_bias)
        q_bg = Normal(z_bg_mean, z_bg_std)
        z_bg = q_bg.rsample()
        bg = self.bg_dec(z_bg)

        for i in range(seq_len):
            x = seq[:, i]
            
            kl_z_what = seq.new_zeros(bs)
            kl_z_where = seq.new_zeros(bs)
            kl_z_depth = seq.new_zeros(bs)
            kl_z_pres = seq.new_zeros(bs)
            kl_edge_type = seq.new_zeros(bs)
            log_imp = seq.new_zeros(bs)
            log = None

            img_enc = img_enc_seq[:, i]

            if i== 0: # we discover object in the first frame
                z_what = x.new_zeros(bs, 2, z_what_dim)
                z_where = x.new_zeros(bs, 2, (z_where_scale_dim + z_where_shift_dim))
                z_where_bias = x.new_zeros(bs, 2, (z_where_scale_dim + z_where_shift_dim))
                z_depth = seq.new_zeros(bs, 2, z_depth_dim)
                z_pres = x.new_zeros(bs, z_pres_dim)
                cumsum_one_minus_z_pres = x.new_zeros(bs, 2, z_pres_dim)
                y_each_obj = x.new_zeros(bs, 2, 3, img_h, img_w)
                alpha_map = x.new_zeros(bs, 2, 1, img_h, img_w)
                importance_map = x.new_zeros(bs, 2, 1, img_h, img_w)
                ids = seq.new_zeros(bs, 2)

                alpha_map_sum = alpha_map.sum(1)
                alpha_map_sum = \
                    alpha_map_sum + (alpha_map_sum.clamp(eps, 1 - eps) - alpha_map_sum).detach()
                
                y_each_obj, alpha_map, importance_map, \
                z_what, z_where, z_where_bias, z_depth, \
                z_pres, ids, kl_z_what, kl_z_where, \
                kl_z_pres, kl_z_depth, log_imp, log = \
                    self.proposal_rejection_cell(
                        x, img_enc, alpha_map_sum, ids, lengths, i, eps=eps
                    )
            else: # propagate in the following frames
                act = actions[:,i-1]
                max_length = max(int(torch.max(lengths)), 2)
                y_each_obj, alpha_map, importance_map, z_what, z_where, \
                z_where_bias, z_depth, z_pres, ids, kl_z_what, kl_z_where, \
                kl_z_depth, kl_z_pres, kl_edge_type, \
                cumsum_one_minus_z_pres, log_imp, log = \
                    self.propagate_cell(
                        x, act, img_enc, 
                        z_what_pre, z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre,
                        cumsum_one_minus_z_pres_pre, ids_pre, lengths, max_length, i, eps=eps
                    )

            importance_map_norm = importance_map / (importance_map.sum(dim=1, keepdim=True) + eps)

            # (bs, 1, img_h, img_w)
            alpha_map_sum = alpha_map.sum(dim=1)
            alpha_map_sum = alpha_map_sum + (alpha_map_sum.clamp(eps, 1 - eps) - alpha_map_sum).detach()
            y_nobg = (y_each_obj.reshape(bs, -1, 3, img_h, img_w) * importance_map_norm).sum(dim=1)

            y = y_nobg + (1 - alpha_map_sum) * bg

            p_x_z = Normal(y.flatten(1), self.args.sigma)
            log_like = p_x_z.log_prob(x.reshape(-1, 3, img_h, img_w).
                                      expand_as(y).flatten(1)).sum(-1)  # sum image dims (C, H, W)
            
            kl_z_pres_all[:, i] = kl_z_pres
            kl_z_what_all[:, i] = kl_z_what
            kl_z_where_all[:, i] = kl_z_where
            kl_z_depth_all[:, i] = kl_z_depth
            kl_edge_type_all[:, i] = kl_edge_type

            if not self.training and self.args.phase_nll:
                log_imp_all[:, i] = log_imp + log_imp
            log_like_all[:, i] = log_like
            y_seq[:, i] = y

            z_mask = (z_pres > self.z_pres_threshold).float()
            cumsum_one_minus_z_pres = seq.new_zeros(bs, z_what.size(1), z_pres_dim)

            num_obj_each = torch.sum(z_mask, dim=1)
            max_num_obj = int(torch.max(num_obj_each).item())
            max_num_obj = max(max_num_obj, 2)

            z_what_pre = seq.new_zeros(bs, max_num_obj, z_what_dim)
            z_where_pre = seq.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
            z_where_bias_pre = seq.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
            z_depth_pre = seq.new_zeros(bs, max_num_obj, z_depth_dim)
            z_pres_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            z_mask_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            cumsum_one_minus_z_pres_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            ids_pre = seq.new_zeros(bs, max_num_obj)

            for b in range(bs):
                num_obj = int(num_obj_each[b])
                idx = z_mask[b].nonzero()[:, 0]
                z_what_pre[b, :num_obj] = z_what[b, idx]
                z_where_pre[b, :num_obj] = z_where[b, idx]
                z_where_bias_pre[b, :num_obj] = z_where_bias[b, idx]
                z_depth_pre[b, :num_obj] = z_depth[b, idx]
                z_pres_pre[b, :num_obj] = z_pres[b, idx]
                z_mask_pre[b, :num_obj] = z_mask[b, idx]
                cumsum_one_minus_z_pres_pre[b, :num_obj] = cumsum_one_minus_z_pres[b, idx]
                ids_pre[b, :num_obj] = ids[b, idx]

            
            if not self.args.phase_do_remove_detach or self.args.global_step < self.args.remove_detach_step:
                z_what_pre = z_what_pre.detach()
                z_where_pre = z_where_pre.detach()
                z_depth_pre = z_depth_pre.detach()
                z_pres_pre = z_pres_pre.detach()
                z_mask_pre = z_mask_pre.detach()
                z_where_bias_pre = z_where_bias_pre.detach()
            
            lengths = torch.sum(z_mask_pre, dim=(1, 2)).reshape(bs)

            scalor_step_log = {}
            if self.args.log_phase:
                #if ids.size(1) < importance_map_norm.size(1):
                #    ids = torch.cat((x.new_zeros(ids[:, 0:2].size()), ids), dim=1)
                id_color = self.color_t[ids.reshape(-1).long() % self.args.color_num]

                # (bs, num_obj + num_cell_h * num_cell_w, 3, 1, 1)
                id_color = id_color.reshape(bs, -1, 3, 1, 1)
                # (bs, num_obj + num_cell_h * num_cell_w, 3, img_h, img_w)
                id_color_map = (alpha_map > .3).float() * id_color
                mask_color = (id_color_map * importance_map_norm.detach()).sum(dim=1)
                x_mask_color = x - 0.7 * (alpha_map_sum > .3).float() * (x - mask_color)
                scalor_step_log = {
                    'y_each_obj': y_each_obj.cpu().detach(),
                    'importance_map_norm': importance_map_norm.cpu().detach(),
                    'importance_map': importance_map.cpu().detach(),
                    'bg': bg.cpu().detach(),
                    'alpha_map': alpha_map_sum.cpu().detach(),
                    'x_mask_color': x_mask_color.cpu().detach(),
                    'mask_color': mask_color.cpu().detach(),
                    'p_bg_what_mean': self.p_bg_what_t1.mean.cpu().detach(),
                    'p_bg_what_std': self.p_bg_what_t1.stddev.cpu().detach(),
                    'z_bg_mean': z_bg_mean.cpu().detach(),
                    'z_bg_std': z_bg_std.cpu().detach()
                }

            scalor_log_list.append(scalor_step_log)
            counting_list.append(lengths)
            
            for k, v in log.items():
                log[k] = v.cpu().detach()
            log_list.append(log)


        # (bs, seq_len)
        counting = torch.stack(counting_list, dim=1)

        log_disc_list = log_list[:1] * seq_len
        log_prop_list = log_list[1:2] + log_list[1:] 

        return y_seq, \
               log_like_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_what_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_where_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_depth_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_pres_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_bg.flatten(start_dim=1).mean(dim=1), \
               kl_edge_type_all.flatten(start_dim=1).mean(dim=1), \
               log_imp_all.flatten(start_dim=1).sum(dim=1), \
               counting, log_disc_list, log_prop_list, scalor_log_list
