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
        self.register_buffer('z_pres_disc_threshold', torch.tensor(.7))
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
        cumsum_one_minus_z_pres_prop_pre = seq.new_zeros(bs, 1, z_pres_dim)
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

        log_disc_list = []
        log_prop_list = []
        scalor_log_list = []
        counting_list = []

        img_enc_seq = self.image_enc(seq.reshape(-1, seq.size(-3), seq.size(-2), seq.size(-1)))
        img_enc_seq = img_enc_seq.reshape(bs, -1, img_enc_seq.size(-3), img_enc_seq.size(-2), img_enc_seq.size(-1))

        self.propagate_cell.node_type = None # wait to initilize inside propagation

        for i in range(seq_len):
            x = seq[:, i]
            
            kl_z_what_prop = seq.new_zeros(bs)
            kl_z_where_prop = seq.new_zeros(bs)
            kl_z_depth_prop = seq.new_zeros(bs)
            kl_z_pres_prop = seq.new_zeros(bs)
            kl_edge_type = seq.new_zeros(bs)
            log_imp_prop = seq.new_zeros(bs)
            log_prop = None

            img_enc = img_enc_seq[:, i]

            if lengths.max() != 0:
                act = actions[:,i-1]
                max_length = max(int(torch.max(lengths)), 2)
                y_each_obj_prop, alpha_map_prop, importance_map_prop, z_what_prop, z_where_prop, \
                z_where_bias_prop, z_depth_prop, z_pres_prop, ids_prop, kl_z_what_prop, kl_z_where_prop, \
                kl_z_depth_prop, kl_z_pres_prop, kl_edge_type, \
                cumsum_one_minus_z_pres_prop, log_imp_prop, log_prop = \
                    self.propagate_cell(
                        x, act, img_enc, 
                        z_what_pre, z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre,
                        cumsum_one_minus_z_pres_prop_pre, ids_pre, lengths, max_length, i, eps=eps
                    )

            else:
                z_what_prop = x.new_zeros(bs, 2, z_what_dim)
                z_where_prop = x.new_zeros(bs, 2, (z_where_scale_dim + z_where_shift_dim))
                z_where_bias_prop = x.new_zeros(bs, 2, (z_where_scale_dim + z_where_shift_dim))
                z_depth_prop = seq.new_zeros(bs, 2, z_depth_dim)
                z_pres_prop = x.new_zeros(bs, z_pres_dim)
                cumsum_one_minus_z_pres_prop = x.new_zeros(bs, 2, z_pres_dim)
                y_each_obj_prop = x.new_zeros(bs, 2, 3, img_h, img_w)
                alpha_map_prop = x.new_zeros(bs, 2, 1, img_h, img_w)
                importance_map_prop = x.new_zeros(bs, 2, 1, img_h, img_w)
                ids_prop = seq.new_zeros(bs, 2)

            alpha_map_prop_sum = alpha_map_prop.sum(1)
            alpha_map_prop_sum = \
                alpha_map_prop_sum + (alpha_map_prop_sum.clamp(eps, 1 - eps) - alpha_map_prop_sum).detach()
            
            y_each_obj_disc, alpha_map_disc, importance_map_disc, \
            z_what_disc, z_where_disc, z_where_bias_disc, z_depth_disc, \
            z_pres_disc, ids_disc, kl_z_what_disc, kl_z_where_disc, \
            kl_z_pres_disc, kl_z_depth_disc, log_imp_disc, log_disc = \
                self.proposal_rejection_cell(
                    x, img_enc, alpha_map_prop_sum, ids_prop, lengths, i, eps=eps
                )

            importance_map = torch.cat((importance_map_prop, importance_map_disc), dim=1)
            importance_map_norm = importance_map / (importance_map.sum(dim=1, keepdim=True) + eps)

            # (bs, 1, img_h, img_w)
            alpha_map = torch.cat((alpha_map_prop, alpha_map_disc), dim=1).sum(dim=1)
            alpha_map = alpha_map + (alpha_map.clamp(eps, 1 - eps) - alpha_map).detach()

            y_each_obj = torch.cat((y_each_obj_prop, y_each_obj_disc), dim=1)
            y_nobg = (y_each_obj.reshape(bs, -1, 3, img_h, img_w) * importance_map_norm).sum(dim=1)

            p_bg_what = self.p_bg_what_t1
            # Background

            z_bg_mean = torch.zeros([bs,bg_what_dim]).to(device)
            z_bg_std = torch.ones([bs,bg_what_dim]).to(device)
            z_bg_std = F.softplus(z_bg_std + self.bg_what_std_bias)
            q_bg = Normal(z_bg_mean, z_bg_std)
            z_bg = q_bg.rsample()
            bg = self.bg_dec(z_bg)

            y = y_nobg + (1 - alpha_map) * bg
            p_x_z = Normal(y.flatten(1), self.args.sigma)
            log_like = p_x_z.log_prob(x.reshape(-1, 3, img_h, img_w).
                                      expand_as(y).flatten(1)).sum(-1)  # sum image dims (C, H, W)

            if not self.args.phase_no_background:
                # Alpha map kl
                try:
                    kl_z_bg[:, i] = 0 #kl_divergence(q_bg, p_bg_what).sum(1)
                except:
                    import pdb; pdb.set_trace()

                ########################################### Compute log importance ############################################
                if not self.training and self.args.phase_nll:
                    # (bs, dim)
                    log_imp_bg = (p_bg_what.log_prob(z_bg) - q_bg.log_prob(z_bg)).sum(1)

                ######################################## End of Compute log importance #########################################
           
            kl_z_pres_all[:, i] = kl_z_pres_disc + kl_z_pres_prop

            kl_z_what_all[:, i] = kl_z_what_disc + kl_z_what_prop
            kl_z_where_all[:, i] = kl_z_where_disc + kl_z_where_prop
            kl_z_depth_all[:, i] = kl_z_depth_disc + kl_z_depth_prop
            kl_edge_type_all[:, i] = kl_edge_type
            if not self.training and self.args.phase_nll:
                log_imp_all[:, i] = log_imp_disc + log_imp_prop + log_imp_bg
            log_like_all[:, i] = log_like
            y_seq[:, i] = y

            if lengths.max() != 0:
                z_what_prop_disc = torch.cat((z_what_prop, z_what_disc), dim=1)
                z_where_prop_disc = torch.cat((z_where_prop, z_where_disc), dim=1)
                z_where_bias_prop_disc = torch.cat((z_where_bias_prop, z_where_bias_disc), dim=1)
                z_depth_prop_disc = torch.cat((z_depth_prop, z_depth_disc), dim=1)
                z_pres_prop_disc = torch.cat((z_pres_prop, z_pres_disc), dim=1)
                z_mask_prop_disc = torch.cat((
                    (z_pres_prop > 0).float(),
                    (z_pres_disc > self.z_pres_disc_threshold+1).float() # do not approve discorvery from 2nd frame
                ), dim=1)
                cumsum_one_minus_z_pres_prop_disc = torch.cat([cumsum_one_minus_z_pres_prop,
                                                               seq.new_zeros(bs, z_what_disc.size(1), z_pres_dim)],
                                                              dim=1)
                ids_prop_disc = torch.cat((ids_prop, ids_disc), dim=1)

            else:
                z_what_prop_disc = z_what_disc
                z_where_prop_disc = z_where_disc
                z_where_bias_prop_disc = z_where_bias_disc
                z_depth_prop_disc = z_depth_disc
                z_pres_prop_disc = z_pres_disc
                z_mask_prop_disc = (z_pres_disc > self.z_pres_disc_threshold).float()
                cumsum_one_minus_z_pres_prop_disc = seq.new_zeros(bs, z_what_disc.size(1), z_pres_dim)
                ids_prop_disc = ids_disc

            num_obj_each = torch.sum(z_mask_prop_disc, dim=1)
            max_num_obj = int(torch.max(num_obj_each).item())

            max_num_obj = max(max_num_obj, 2)

            z_what_pre = seq.new_zeros(bs, max_num_obj, z_what_dim)
            z_where_pre = seq.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
            z_where_bias_pre = seq.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
            z_depth_pre = seq.new_zeros(bs, max_num_obj, z_depth_dim)
            z_pres_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            z_mask_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            cumsum_one_minus_z_pres_prop_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            ids_pre = seq.new_zeros(bs, max_num_obj)

            for b in range(bs):
                num_obj = int(num_obj_each[b])

                idx = z_mask_prop_disc[b].nonzero()[:, 0]

                z_what_pre[b, :num_obj] = z_what_prop_disc[b, idx]
                z_where_pre[b, :num_obj] = z_where_prop_disc[b, idx]
                z_where_bias_pre[b, :num_obj] = z_where_bias_prop_disc[b, idx]
                z_depth_pre[b, :num_obj] = z_depth_prop_disc[b, idx]
                z_pres_pre[b, :num_obj] = z_pres_prop_disc[b, idx]
                z_mask_pre[b, :num_obj] = z_mask_prop_disc[b, idx]
                cumsum_one_minus_z_pres_prop_pre[b, :num_obj] = cumsum_one_minus_z_pres_prop_disc[b, idx]
                ids_pre[b, :num_obj] = ids_prop_disc[b, idx]

            '''
            if not self.args.phase_do_remove_detach or self.args.global_step < self.args.remove_detach_step:
                z_what_pre = z_what_pre.detach()
                z_where_pre = z_where_pre.detach()
                z_depth_pre = z_depth_pre.detach()
                z_pres_pre = z_pres_pre.detach()
                z_mask_pre = z_mask_pre.detach()
                z_where_bias_pre = z_where_bias_pre.detach()
                print('haha')
            '''
            
            lengths = torch.sum(z_mask_pre, dim=(1, 2)).reshape(bs)

            scalor_step_log = {}
            if self.args.log_phase:
                if ids_prop_disc.size(1) < importance_map_norm.size(1):
                    ids_prop_disc = torch.cat((x.new_zeros(ids_prop_disc[:, 0:2].size()), ids_prop_disc), dim=1)
                id_color = self.color_t[ids_prop_disc.reshape(-1).long() % self.args.color_num]

                # (bs, num_obj_prop + num_cell_h * num_cell_w, 3, 1, 1)
                id_color = id_color.reshape(bs, -1, 3, 1, 1)
                # (bs, num_obj_prop + num_cell_h * num_cell_w, 3, img_h, img_w)
                id_color_map = (torch.cat((alpha_map_prop, alpha_map_disc), dim=1) > .3).float() * id_color
                mask_color = (id_color_map * importance_map_norm.detach()).sum(dim=1)
                x_mask_color = x - 0.7 * (alpha_map > .3).float() * (x - mask_color)
                scalor_step_log = {
                    'y_each_obj': y_each_obj.cpu().detach(),
                    'importance_map_norm': importance_map_norm.cpu().detach(),
                    'importance_map': importance_map.cpu().detach(),
                    'bg': bg.cpu().detach(),
                    'alpha_map': alpha_map.cpu().detach(),
                    'x_mask_color': x_mask_color.cpu().detach(),
                    'mask_color': mask_color.cpu().detach(),
                    'p_bg_what_mean': self.p_bg_what_t1.mean.cpu().detach(),
                    'p_bg_what_std': self.p_bg_what_t1.stddev.cpu().detach(),
                    'z_bg_mean': z_bg_mean.cpu().detach(),
                    'z_bg_std': z_bg_std.cpu().detach()
                }

                if log_disc:
                    for k, v in log_disc.items():
                        log_disc[k] = v.cpu().detach()
                if log_prop:
                    for k, v in log_prop.items():
                        log_prop[k] = v.cpu().detach()

            log_disc_list.append(log_disc)
            log_prop_list.append(log_prop)
            scalor_log_list.append(scalor_step_log)
            counting_list.append(lengths)

        # (bs, seq_len)
        counting = torch.stack(counting_list, dim=1)

        frame_to_train = random.randint(0, seq_len-1)
        #log_like_all[:,frame_to_train:frame_to_train+1].flatten(start_dim=1).mean(dim=1), \

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
