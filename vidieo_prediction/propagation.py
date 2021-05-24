import numpy as np
import torch
from torch.distributions import Normal, kl_divergence, Bernoulli, RelaxedBernoulli
from torch import nn
import torch.nn.functional as F
from utils import spatial_transform, calc_kl_z_pres_bernoulli, calc_kl_z_edge_bernoulli
from common import *
from vision_modules import NumericalRelaxedBernoulli, gumbel_softmax, HLoss
from graph_modules import PropNet 

criterionH = HLoss()

class PropagationCell(nn.Module):

    def __init__(self, args, z_what_net, glimpse_dec_net):
        super(PropagationCell, self).__init__()
        self.args = args

        self.z_pres_logits_bias = 2.
        self.where_update_scale = where_update_scale
        self.z_where_std_bias = -2

        # self.z_what_gate_bias = 2
        self.register_buffer('z_pres_stop_threshold', torch.tensor(0.6))

        # z where nets
        z_where_transit_bias_net_input_dim = nf + z_what_dim + z_where_scale_dim + \
                                             z_where_shift_dim + z_where_bias_dim
        self.z_where_transit_bias_net = nn.Sequential(
            nn.Linear(z_where_transit_bias_net_input_dim, nf),
            nn.CELU(),
            #nn.BatchNorm1d(nf),
            nn.Linear(nf, (z_where_scale_dim + z_where_shift_dim) * 2)
        )

        # z depth net
        z_depth_transit_net_input_dim = nf + z_what_dim + z_depth_dim

        self.z_depth_transit_net = nn.Sequential(
            nn.Linear(z_depth_transit_net_input_dim, nf),
            nn.CELU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, z_depth_dim * 2)
        )

        # z what net 
        self.z_what_from_transit_net = nn.Sequential(
            nn.Linear(nf, nf),
            nn.CELU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, z_what_dim * 2)
        )

        # z pres net 
        z_pres_transit_input_dim = nf + z_where_scale_dim + \
                                   z_where_shift_dim + z_where_bias_dim + z_what_dim

        self.z_pres_transit = nn.Sequential(
            nn.Linear(z_pres_transit_input_dim, nf),
            nn.CELU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, z_pres_dim),
        )

        infer_graph_struct_node_dim = z_where_scale_dim + z_where_shift_dim + \
                                z_pres_dim + z_what_dim + z_where_bias_dim + z_depth_dim
        # infer node type
        self.infer_node_type = PropNet(
            node_dim_in=z_what_dim,
            edge_dim_in=0,
            nf_hidden=nf * 3,
            node_dim_out=1,
            edge_dim_out=0,
            edge_type_num=1,
            pstep=0,
            batch_norm=1)

        # infer edge type
        self.infer_edge_type = PropNet(
            node_dim_in=infer_graph_struct_node_dim,
            edge_dim_in=0,
            nf_hidden=nf * 3,
            node_dim_out=0,
            edge_dim_out=2,
            edge_type_num=1,
            pstep=0,
            batch_norm=1)


        # object transit graph nets
        object_transit_inp_dim = z_where_scale_dim + z_where_shift_dim + z_pres_dim + \
                                z_what_dim + z_where_bias_dim + z_depth_dim + action_dim

        self.object_transit_net = PropNet(
            node_dim_in=object_transit_inp_dim,
            edge_dim_in=0,
            nf_hidden=nf * 3,
            node_dim_out=nf,
            edge_dim_out=nf,
            edge_type_num=2,
            pstep=2, # no batchnorm and no propagation in!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            batch_norm=1)

        self.object_transit_mlp_net = nn.Sequential(
            nn.Linear(object_transit_inp_dim, nf),
            nn.CELU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
            nn.CELU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
            nn.CELU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
        )

        self.glimpse_dec_net = glimpse_dec_net
        self.z_what_net = z_what_net
        self.node_type = None

        z_what_gate_net_inp_dim = nf

        self.z_what_gate_net = nn.Sequential(
            nn.Linear(z_what_gate_net_inp_dim, 64),
            nn.CELU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x, act, img_enc, 
                z_what_pre, z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre,
                cumsum_one_minus_z_pres, ids_pre, lengths, max_length, t, eps=1e-15):
        """

        :param x: input image (bs, c, h, w)
        :param act: action (bs, act_dim)
        :param img_enc: input image encode (bs, c, num_cell_h, num_cell_w)
        :param z_what_pre: (bs, max_num_obj, dim)
        :param z_where_pre: (bs, max_num_obj, dim)
        :param z_depth_pre: (bs, max_num_obj, dim)
        :param z_pres_pre: (bs, max_num_obj, dim)
        :param cumsum_one_minus_z_pres: (bs, max_num_obj, dim)
        :param lengths: (bs)
        :return:
        """
        '''
        We zero img infos to push model not using info there
        '''
        #x = x * 0

        bs = x.size(0)
        device = x.device
        max_num_obj = max_length
        obj_mask = (z_pres_pre.view(bs, max_num_obj) != 0).float()

        bns = bs * max_num_obj

        # node_rep: B x N x infer_graph_struct_node_dims
        obj_rep = torch.cat(
            [z_where_pre, z_pres_pre, z_what_pre, z_where_bias_pre, z_depth_pre],
            dim=2
        )

        if self.node_type == None:
            # node_type_logits: bs x max_num_obj x 2
            node_type_logits = self.infer_node_type(node_rep=z_what_pre, ignore_edge=True)
            # node_type: bs x max_num_obj
            self.node_type = gumbel_softmax(node_type_logits.view(bs, max_num_obj), hard=hard_gumble_softmax)

        node_type = self.node_type
        # expanded_action: bs x max_num_obj x action_dim
        expanded_node_type = self.node_type.unsqueeze(2).expand(-1,-1,action_dim).contiguous().to(device)
        expanded_action = act.unsqueeze(1).expand(-1,max_num_obj,-1).contiguous().to(device)
        
        # edge_type_logits: bs x max_num_obj x 2    
        edge_type_logits = self.infer_edge_type(node_rep=obj_rep, ignore_node=True)

        if edge_share:
            edge_type_logits = (edge_type_logits + torch.transpose(edge_type_logits, 1, 2)) / 2.
       
        # edge_type: (bs * max_num_obj * max_num_obj) x 2       
        edge_type = gumbel_softmax(edge_type_logits.view(bs * max_num_obj * max_num_obj, 2), hard=hard_gumble_softmax)
        expanded_action = expanded_action * expanded_node_type

        obj_act_inp = torch.cat(
            [z_where_pre, z_pres_pre, z_what_pre, z_where_bias_pre, z_depth_pre, expanded_action],
            dim=2
        )
        object_transit_out = self.object_transit_net(
            obj_act_inp,
            None,
            edge_type,
            start_idx=edge_st_idx,
            ignore_edge=True)

        object_transit_out = self.object_transit_mlp_net(obj_act_inp.view(bns, -1).contiguous()).view(bs,max_num_obj,-1)
        #import pdb; pdb.set_trace()

        # z_where transition
        z_where_transit_bias_net_inp = torch.cat(
            [object_transit_out, z_what_pre, z_where_pre, z_where_bias_pre], dim=2
        )
        # bns x dim
        z_where_transit_bias_net_inp = z_where_transit_bias_net_inp.view(bns, -1).contiguous()

        object_transit_out = object_transit_out.view(bns, -1).contiguous()
       
        z_where_bias_mean, z_where_bias_std = \
            self.z_where_transit_bias_net(z_where_transit_bias_net_inp).chunk(2, -1)
        z_where_bias_std = F.softplus(z_where_bias_std + self.z_where_std_bias)
        z_where_bias_dist = Normal(z_where_bias_mean, z_where_bias_std)
        z_where_bias = z_where_bias_dist.rsample()

        z_where_pre = z_where_pre.view(bns, -1).contiguous()
        z_where_shift = z_where_pre[:, 2:] + self.where_update_scale * z_where_bias[:, 2:].tanh()

        scale, ratio = z_where_bias[:, :2].tanh().chunk(2, 1)
        scale = self.args.size_anc + self.args.var_s * scale  # add bias to let masking do its job
        ratio = self.args.ratio_anc + self.args.var_anc * ratio
        ratio_sqrt = ratio.sqrt()

        z_where = torch.cat((scale / ratio_sqrt, scale * ratio_sqrt, z_where_shift), dim=1)
        # # always within the image
        z_where = torch.cat((z_where[:, :2], z_where[:, 2:].clamp(-1.05, 1.05)), dim=1)

        # z_what transit
        # encode
        x_att = \
            spatial_transform(
                x.unsqueeze(1).expand(-1, max_num_obj, -1, -1, -1).contiguous().view(bns, 3, img_h, img_w), z_where,
                (bns, 3, glimpse_size, glimpse_size), inverse=False
            )

        z_what_from_enc_mean, z_what_from_enc_std = self.z_what_net(
            x_att
        )
        z_what_from_enc_std = F.softplus(z_what_from_enc_std)
        z_what_encode_dist = Normal(z_what_from_enc_mean, z_what_from_enc_std)

        # transit
        z_what_from_transit_mean, z_what_from_transit_std = \
            self.z_what_from_transit_net(object_transit_out).chunk(2, -1)

        z_what_from_transit_std = F.softplus(z_what_from_transit_std)
        z_what_transit_dist = Normal(z_what_from_transit_mean, z_what_from_transit_std)

        if True or self.args.phase_generate and t >= self.args.observe_frames:
            z_what_mean = z_what_from_transit_mean
            z_what_std = z_what_from_transit_std
            z_what_dist = z_what_transit_dist
        else:
            z_what_gate_net_inp = object_transit_out
            forget_gate, input_gate = self.z_what_gate_net(z_what_gate_net_inp).chunk(2, -1)
            z_what_mean = input_gate * z_what_from_enc_mean + \
                      forget_gate * z_what_from_transit_mean

            z_what_std = F.softplus(input_gate * z_what_from_enc_std + \
                                forget_gate * z_what_from_transit_std)

            z_what_dist = Normal(z_what_mean, z_what_std)

        z_what = z_what_dist.rsample()
       

        # z depth transit
        z_depth_pre = z_depth_pre.view(bns, -1).contiguous()
        z_depth_transit_net_inp = torch.cat(
            [object_transit_out, z_what, z_depth_pre],
            dim=1
        )
        z_depth_mean, z_depth_std = self.z_depth_transit_net(z_depth_transit_net_inp).chunk(2, -1)
        z_depth_std = F.softplus(z_depth_std)
        z_depth_dist = Normal(z_depth_mean, z_depth_std)
        z_depth = z_depth_dist.rsample()

        # z_pres bns, dim
        z_pres_transit_inp = torch.cat(
            [object_transit_out, z_where, z_where_bias, z_what],
            dim=1
        )
        z_pres_logits = pres_logit_factor * torch.tanh(self.z_pres_transit(z_pres_transit_inp) +
                                                       self.z_pres_logits_bias)

        z_pres_dist = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=self.args.tau)
        z_pres_y = z_pres_dist.rsample()
        #z_pres = torch.sigmoid(z_pres_y+1000)
        z_pres = torch.ones(z_pres_y.shape).to(device) # make it all 1 for now

        o_att, alpha_att = self.glimpse_dec_net(z_what)

        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)
        y_att = alpha_att_hat * o_att

        # (bs, 3, img_h, img_w)
        y_each_obj = spatial_transform(y_att, z_where, (bns, 3, img_h, img_w), inverse=True)

        # (batch_size_t, 1, glimpse_size, glimpse_size)
        importance_map = alpha_att_hat * torch.sigmoid(-z_depth).view(-1, 1, 1, 1)
        #import pdb; pdb.set_trace()
        # (batch_size_t, 1, img_h, img_w)
        importance_map_full_res = spatial_transform(importance_map, z_where, (bns, 1, img_h, img_w),
                                                    inverse=True)

        # (batch_size_t, 1, img_h, img_w)
        alpha_map = spatial_transform(alpha_att_hat, z_where, (bns, 1, img_h, img_w), inverse=True)
        final_z_pres_mask = z_pres.squeeze() * obj_mask.view(bns)

        kl_z_pres = torch.zeros(bs).to(device)
        kl_z_what = \
            (kl_divergence(z_what_transit_dist, z_what_encode_dist).sum(1) * \
             z_pres.squeeze() * obj_mask.view(bns)).view(bs, max_num_obj).sum(1)        
        kl_z_where = torch.zeros(bs).to(device)
        kl_z_depth = torch.zeros(bs).to(device)

        #pres_edge_type_logits = torch.index_select(edge_type_logits.view(-1,2), 0, (final_z_pres_mask == 1).nonzero().squeeze())
        #edge_type_prior = torch.FloatTensor(np.array([1-edge_pos_prior, edge_pos_prior])).cuda()
        #kl_edge_type = -criterionH(pres_edge_type_logits.view(-1,2), edge_type_prior) 
        tmp_obj_mask = obj_mask.view(bs, max_num_obj)
        edge_mask = torch.bmm(tmp_obj_mask.unsqueeze(2), tmp_obj_mask.unsqueeze(1)).view(-1)

        kl_edge_type = \
            (calc_kl_z_edge_bernoulli(edge_type_logits.view(-1,2), torch.tensor(edge_pos_prior)) * 
                edge_mask.view(-1)).view(bs, max_num_obj * max_num_obj).sum(1)
        ########################################### Compute log importance ############################################
        log_imp = x.new_zeros(bs)
        if not self.training and self.args.phase_nll:
            z_pres_binary = (z_pres > 0.5).float()
            # (bns, dim)
            log_imp = torch.zeros(bs,1).to(device)

        ######################################## End of Compute log importance #########################################
        z_what_all = z_what.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        z_where_dummy = x.new_ones(bs, max_num_obj, (z_where_scale_dim + z_where_shift_dim)) * .5
        z_where_dummy[:, :, z_where_scale_dim:] = 2
        z_where_all = z_where.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1) + \
                      z_where_dummy * (1 - obj_mask.view(bs, max_num_obj, 1))
        z_where_bias_all = z_where_bias.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        z_pres_all = z_pres.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)

        z_depth_all = z_depth.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        y_each_obj_all = \
            y_each_obj.view(bs, max_num_obj, 3, img_h, img_w) * obj_mask.view(bs, max_num_obj, 1, 1, 1)
        alpha_map_all = \
            alpha_map.view(bs, max_num_obj, 1, img_h, img_w) * obj_mask.view(bs, max_num_obj, 1, 1, 1)
        importance_map_all = \
            importance_map_full_res.view(bs, max_num_obj, 1, img_h, img_w) * \
            obj_mask.view(bs, max_num_obj, 1, 1, 1)

        cumsum_one_minus_z_pres = cumsum_one_minus_z_pres.view(bs, max_num_obj, -1)

        if self.args.log_phase:
            self.log = {
                'z_what': z_what_all,
                'z_where': z_where_all,
                'z_pres': z_pres_all,
                'z_what_std': z_what_std.view(bs, max_num_obj, -1),
                'z_what_mean': z_what_mean.view(bs, max_num_obj, -1),
                'z_where_bias_std': z_where_bias_std.view(bs, max_num_obj, -1),
                'z_where_bias_mean': z_where_bias_mean.view(bs, max_num_obj, -1),
                'lengths': lengths,
                'z_depth': z_depth_all,
                'z_depth_std': z_depth_std.view(bs, max_num_obj, -1),
                'z_depth_mean': z_depth_mean.view(bs, max_num_obj, -1),
                'y_each_obj': y_each_obj_all.view(bs, max_num_obj, 3, img_h, img_w),
                'alpha_map': alpha_map_all.view(bs, max_num_obj, 1, img_h, img_w),
                'importance_map': importance_map_all.view(bs, max_num_obj, 1, img_h, img_w),
                'z_pres_logits': z_pres_logits.view(bs, max_num_obj, -1),
                'z_pres_y': z_pres_y.view(bs, max_num_obj, -1),
                'o_att': o_att.view(bs, max_num_obj, 3, glimpse_size, glimpse_size),
                'z_where_bias': z_where_bias_all,
                'node_type': node_type,
                'edge_type': edge_type,
                'ids': ids_pre
            }
        else:
            self.log = {}
        #print(z_pres_all)
        return y_each_obj_all, alpha_map_all, importance_map_all, z_what_all, z_where_all, \
               z_where_bias_all, z_depth_all, z_pres_all, ids_pre, kl_z_what, kl_z_where, kl_z_depth, \
               kl_z_pres, kl_edge_type, cumsum_one_minus_z_pres, log_imp, self.log
