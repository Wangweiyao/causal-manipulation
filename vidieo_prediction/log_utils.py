import torch
from copy import copy
from torchvision.utils import make_grid
from utils import spatial_transform, visualize
from common import *
from skimage.draw import disk, rectangle, line_aa
import cv2
import numpy as np
from copy import deepcopy

blue = (66, 133, 244) 
red = (219, 68, 65)
yellow = (244, 180, 0)
green = (15, 157, 88)

def draw_graph(keypoint, node_type, edge_type, lim, c):
    # draw pred confidence
    fig, ax = plt.subplots(1)
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])

    height = 400.

    for j in range(keypoint.shape[0]):
        x, y = keypoint[j, 0], keypoint[j, 1]
        x = x / height * 2
        x -= lim[1]
        y = y / height * 2
        y -= lim[3]
        y = -y

        for k in range(keypoint.shape[0]):
            if k == j:
                continue
            xx, yy = keypoint[k, 0], keypoint[k, 1]
            xx = xx / height * 2
            xx -= lim[1]
            yy = yy / height * 2
            yy -= lim[3]
            yy = -yy

            edge_type_cur = edge_type[j, k]
            if edge_type_cur < edge_st_idx:
                continue

            dist = norm(np.array([x - xx, y - yy]))

            direct = np.array([x - xx, y - yy]) / dist
            ax.arrow(xx + direct[0] * 0.05, yy + direct[1] * 0.05,
                        x - xx - direct[0] * 0.15, y - yy - direct[1] * 0.15,
                        fc=c[edge_type_cur], ec='w', width=0.02, head_width=0.06, head_length=0.06, alpha=0.5)

        ax.scatter(x, y, c=c[j], s=150)

    ax.set_aspect('equal')
    plt.tight_layout()

    # plt.show()
    plt.savefig(file_name)

    plt.close()


def log_summary(args, writer, imgs, y_seq, global_step, log_disc_list,
                log_prop_list, scalor_log_list, prefix='train', eps=1e-15):
    args = copy(args)
    if prefix == 'test':
        args.num_img_summary = args.num_img_summary
    bs = imgs.size(0)
    seq_len = imgs.shape[1]
    grid_image = make_grid(imgs[:args.num_img_summary * 2].cpu().view(-1, 3, img_h, img_w), seq_len,
                           normalize=True,
                           pad_value=1)
    writer.add_image(f'{prefix}_scalor/1-image', grid_image, global_step)

    grid_image = make_grid(y_seq[:args.num_img_summary * 2].cpu().view(-1, 3, img_h, img_w), seq_len,
                           normalize=True,
                           pad_value=1)
    writer.add_image(f'{prefix}_scalor/2-reconstruction_overall', grid_image, global_step)

    bbox_prop_list = []
    bbox_disc_list = []
    recon_prop_list = []
    recon_disc_list = []
    bg_list = []
    alpha_map_list = []
    x_mask_color_list = []
    # for each time step
    for j in range(imgs.size(1)):
        
        # first recon from disc and recon from
        y_each_obj = scalor_log_list[j]['y_each_obj'][:args.num_img_summary]
        importance_map_norm = scalor_log_list[j]['importance_map_norm'][:args.num_img_summary]

        y_prop_disc = y_each_obj * importance_map_norm

        recon_prop_list.append(y_prop_disc[:, :-args.num_cell_h * args.num_cell_w].sum(dim=1))
        recon_disc_list.append(y_prop_disc[:, -args.num_cell_h * args.num_cell_w:].sum(dim=1))
        bg_list.append(scalor_log_list[j]['bg'][:args.num_img_summary])

        alpha_map_list.append(scalor_log_list[j]['alpha_map'][:args.num_img_summary])
        x_mask_color_list.append(scalor_log_list[j]['x_mask_color'][:args.num_img_summary])

        if prefix == 'train' and not args.phase_simplify_summary:
            writer.add_histogram(f'{prefix}_inside_value_scalor_{j}/importance_map_norm',
                                 scalor_log_list[j]['importance_map_norm']
                                 [scalor_log_list[j]['importance_map_norm'] > 0].cpu().detach().numpy(),
                                 global_step)
            for k, v in scalor_log_list[j].items():
                if '_bg_' in k:
                    writer.add_histogram(f'{prefix}_inside_value_scalor_{j}/{k}', v.cpu().detach().numpy(), global_step)
            if args.phase_conv_lstm:
                for k, v in scalor_log_list[j].items():
                    if 'lstm' in k:
                        writer.add_histogram(f'{prefix}_inside_value_scalor_{j}/{k}', v.cpu().detach().numpy(),
                                             global_step)

        num_cell_h, num_cell_w = cfg['num_cell_h'], cfg['num_cell_w']
        
        log_disc = {
            'z_what': log_disc_list[j]['z_what'].view(-1, num_cell_h * num_cell_w, z_what_dim),
            'z_where_scale':
                log_disc_list[j]['z_where'].view(-1, num_cell_h * num_cell_w, z_where_scale_dim + z_where_shift_dim)[:, :,
                :z_where_scale_dim],
            'z_where_shift':
                log_disc_list[j]['z_where'].view(-1, num_cell_h * num_cell_w, z_where_scale_dim + z_where_shift_dim)[:, :,
                z_where_scale_dim:],
            'z_pres': log_disc_list[j]['z_pres'].permute(0, 2, 3, 1),
            'z_pres_probs': torch.sigmoid(log_disc_list[j]['z_pres_logits']).permute(0, 2, 3, 1),
            'z_what_std': log_disc_list[j]['z_what_std'].view(-1, num_cell_h * num_cell_w, z_what_dim),
            'z_what_mean': log_disc_list[j]['z_what_mean'].view(-1, num_cell_h * num_cell_w, z_what_dim),
            'z_where_scale_std':
                log_disc_list[j]['z_where_std'].permute(0, 2, 3, 1)[:, :, :z_where_scale_dim],
            'z_where_scale_mean':
                log_disc_list[j]['z_where_mean'].permute(0, 2, 3, 1)[:, :, :z_where_scale_dim],
            'z_where_shift_std':
                log_disc_list[j]['z_where_std'].permute(0, 2, 3, 1)[:, :, z_where_scale_dim:],
            'z_where_shift_mean':
                log_disc_list[j]['z_where_mean'].permute(0, 2, 3, 1)[:, :, z_where_scale_dim:],
            'glimpse': log_disc_list[j]['x_att'].view(-1, num_cell_h * num_cell_w, 3, glimpse_size, glimpse_size) \
                if prefix != 'generate' else None,
            'glimpse_recon': log_disc_list[j]['y_att'].view(-1, num_cell_h * num_cell_w, 3, glimpse_size, glimpse_size),
            'prior_z_pres_prob': log_disc_list[j]['prior_z_pres_prob'].unsqueeze(0),
            'o_each_cell': spatial_transform(log_disc_list[j]['o_att'], log_disc_list[j]['z_where'],
                                             (num_cell_h * num_cell_w * bs, 3, img_h, img_w),
                                             inverse=True).view(-1, num_cell_h * num_cell_w, 3, img_h, img_w),
            'alpha_hat_each_cell': spatial_transform(log_disc_list[j]['alpha_att_hat'],
                                                     log_disc_list[j]['z_where'],
                                                     (num_cell_h * num_cell_w * bs, 1, img_h, img_w),
                                                     inverse=True).view(-1, num_cell_h * num_cell_w, 1, img_h, img_w),
            'alpha_each_cell': spatial_transform(log_disc_list[j]['alpha_att'], log_disc_list[j]['z_where'],
                                                 (num_cell_h * num_cell_w * bs, 1, img_h, img_w),
                                                 inverse=True).view(-1, num_cell_h * num_cell_w, 1, img_h, img_w),
            'y_each_cell': (log_disc_list[j]['y_each_cell'] * log_disc_list[j]['z_pres'].
                            view(-1, 1, 1, 1)).view(-1, num_cell_h * num_cell_w, 3, img_h, img_w),
            'z_depth': log_disc_list[j]['z_depth'].view(-1, num_cell_h * num_cell_w, z_depth_dim),
            'z_depth_std': log_disc_list[j]['z_depth_std'].view(-1, num_cell_h * num_cell_w, z_depth_dim),
            'z_depth_mean': log_disc_list[j]['z_depth_mean'].view(-1, num_cell_h * num_cell_w, z_depth_dim),
            'z_pres_logits': log_disc_list[j]['z_pres_logits'].permute(0, 2, 3, 1),
            'z_pres_y': log_disc_list[j]['z_pres_y'].permute(0, 2, 3, 1)
        }

        bbox = visualize(imgs[:args.num_img_summary, j].cpu(),
                         log_disc['z_pres'][:args.num_img_summary].cpu().detach(),
                         log_disc['z_where_scale'][:args.num_img_summary].cpu().detach(),
                         log_disc['z_where_shift'][:args.num_img_summary].cpu().detach())

        y_each_cell = log_disc['y_each_cell'].view(-1, 3, img_h, img_w)[
                      :args.num_img_summary * args.num_cell_h * args.num_cell_w].cpu().detach()
        o_each_cell = log_disc['o_each_cell'].view(-1, 3, img_h, img_w)[
                      :args.num_img_summary * args.num_cell_h * args.num_cell_w].cpu().detach()
        alpha_each_cell = log_disc['alpha_hat_each_cell'].view(-1, 1, img_h, img_w)[
                          :args.num_img_summary * args.num_cell_h * args.num_cell_w].cpu().detach()

        if log_prop_list[j]:
            log_prop = {
                'z_what': log_prop_list[j]['z_what'].view(bs, -1, z_what_dim),
                'z_where_scale':
                    log_prop_list[j]['z_where'].view(bs, -1, z_where_scale_dim + z_where_shift_dim)[:, :,
                    :z_where_scale_dim],
                'z_where_shift':
                    log_prop_list[j]['z_where'].view(bs, -1, z_where_scale_dim + z_where_shift_dim)[:, :,
                    z_where_scale_dim:],
                'z_pres': log_prop_list[j]['z_pres'],
                'z_what_std': log_prop_list[j]['z_what_std'].view(bs, -1, z_what_dim),
                'z_what_mean': log_prop_list[j]['z_what_mean'].view(bs, -1, z_what_dim),
                'z_where_bias_scale_std':
                    log_prop_list[j]['z_where_bias_std'][:, :, :z_where_scale_dim],
                'z_where_bias_scale_mean':
                    log_prop_list[j]['z_where_bias_mean'][:, :, :z_where_scale_dim],
                'z_where_bias_shift_std':
                    log_prop_list[j]['z_where_bias_std'][:, :, z_where_scale_dim:],
                'z_where_bias_shift_mean':
                    log_prop_list[j]['z_where_bias_mean'][:, :, z_where_scale_dim:],
                'z_pres_probs': torch.sigmoid(log_prop_list[j]['z_pres_logits']),
                #'glimpse': log_prop_list[j]['glimpse'],
                #'glimpse_recon': log_prop_list[j]['glimpse_recon'],
                #'prior_z_pres_prob': log_prop_list[j]['prior_z_pres_prob'],
                #'prior_where_bias_scale_std':
                #    log_prop_list[j]['prior_where_bias_std'][:, :, :z_where_scale_dim],
                #'prior_where_bias_scale_mean':
                #    log_prop_list[j]['prior_where_bias_mean'][:, :, :z_where_scale_dim],
                #'prior_where_bias_shift_std':
                #    log_prop_list[j]['prior_where_bias_std'][:, :, z_where_scale_dim:],
                #'prior_where_bias_shift_mean':
                #    log_prop_list[j]['prior_where_bias_mean'][:, :, z_where_scale_dim:],
                'lengths': log_prop_list[j]['lengths'],
                'z_depth': log_prop_list[j]['z_depth'],
                'z_depth_std': log_prop_list[j]['z_depth_std'],
                'z_depth_mean': log_prop_list[j]['z_depth_mean'],

                'y_each_obj': log_prop_list[j]['y_each_obj'],
                'alpha_hat_each_obj': log_prop_list[j]['alpha_map'],

                'z_pres_logits': log_prop_list[j]['z_pres_logits'],
                'z_pres_y': log_prop_list[j]['z_pres_y'],
                'o_each_obj':
                    spatial_transform(log_prop_list[j]['o_att'].view(-1, 3, glimpse_size, glimpse_size),
                                      log_prop_list[j]['z_where'].view(-1, (z_where_scale_dim +
                                                                            z_where_shift_dim)),
                                      (log_prop_list[j]['o_att'].size(1) * bs, 3, img_h, img_w),
                                      inverse=True).view(bs, -1, 3, img_h, img_w),
                'z_where_bias_scale':
                    log_prop_list[j]['z_where_bias'].view(bs, -1, z_where_scale_dim + z_where_shift_dim)
                    [:, :, :z_where_scale_dim],
                'z_where_bias_shift':
                    log_prop_list[j]['z_where_bias'].view(bs, -1, z_where_scale_dim + z_where_shift_dim)
                    [:, :, z_where_scale_dim:],
            }

            num_obj = log_prop['z_pres'].size(1)
            idx = [[], []]
            for k in range(bs):
                for l in range(int(log_prop['lengths'][k])):
                    idx[0].append(k)
                    idx[1].append(l)
            idx_false = [[], []]
            for k in range(bs):
                for l in range(num_obj - int(log_prop['lengths'][k])):
                    idx_false[0].append(k)
                    idx_false[1].append(int(log_prop['lengths'][k] + l))
            if prefix == 'train' and not args.phase_simplify_summary:
                for key, value in log_prop.items():
                    if key == 'lengths':
                        writer.add_histogram(f'{prefix}_inside_value_prop_{j}/{key}', value.cpu().detach().numpy(),
                                             global_step)
                    else:
                        writer.add_histogram(f'{prefix}_inside_value_prop_{j}/{key}', value.cpu().detach()[idx].numpy(),
                                             global_step)

            bbox_prop = visualize(imgs[:args.num_img_summary, j].cpu(),
                                  log_prop['z_pres'][:args.num_img_summary].cpu().detach(),
                                  log_prop['z_where_scale'][:args.num_img_summary].cpu().detach(),
                                  log_prop['z_where_shift'][:args.num_img_summary].cpu().detach(),
                                  only_bbox=True)

            bbox_prop = bbox_prop.view(args.num_img_summary, -1, 3, img_h, img_w)
            bbox_prop_one_time_step = (bbox_prop.sum(dim=1) + imgs[:args.num_img_summary, j].cpu()).clamp(0, 1)
            bbox_prop_list.append(bbox_prop_one_time_step)

            node_type = log_prop_list[j]['node_type']
            edge_type = log_prop_list[j]['edge_type'].reshape(log_prop_list[j]['node_type'].shape[0], log_prop_list[j]['node_type'].shape[1], log_prop_list[j]['node_type'].shape[1],2)
            
            back_imgs = bg_list[-1]
            for idx_bg in range(back_imgs.shape[0]):
                img_cv2_format = np.uint8(255*(back_imgs[idx_bg].cpu().numpy()).transpose(1,2,0)).copy()
                pres_idxs = [i for i, e in enumerate(log_prop['z_pres'][idx_bg,:,0]) if e == 1]
                for idx_node in pres_idxs:
                    x_c = log_prop['z_where_shift'][idx_bg,idx_node,0].cpu().numpy() * 32 + 32
                    y_c = log_prop['z_where_shift'][idx_bg,idx_node,1].cpu().numpy() * 32 + 32
                    center_coordinates = (int(x_c), int(y_c))

                    if node_type[idx_bg,idx_node] > 0.5:
                        color = red
                    else:
                        color = blue

                    for idx_node_2 in pres_idxs:
                        if idx_node != idx_node_2:
                            x_c_2 = int(log_prop['z_where_shift'][idx_bg,idx_node_2,0].cpu().numpy() * 32 + 32)
                            y_c_2 = int(log_prop['z_where_shift'][idx_bg,idx_node_2,1].cpu().numpy() * 32 + 32)

                            if edge_type[idx_bg,idx_node,idx_node_2,1] > 0.5:
                                img_cv2_format = cv2.line(img_cv2_format, center_coordinates, (x_c_2, y_c_2), green, 1, lineType=cv2.LINE_AA)

                    img_cv2_format = cv2.circle(img_cv2_format, center_coordinates, 3, color, -1, lineType=cv2.LINE_AA)

                back_imgs[idx_bg] = torch.tensor(img_cv2_format.transpose(2,0,1) / 255)

            #
        else:
            bbox_prop_one_time_step = imgs[:args.num_img_summary, j].cpu()
            bbox_prop_list.append(bbox_prop_one_time_step)
        if prefix == 'train' and not args.phase_simplify_summary:
            for key, value in log_disc.items():
                writer.add_histogram(f'{prefix}_inside_value_disc_{j}/{key}', value.cpu().detach().numpy(),
                                     global_step)
        
        if not args.phase_simplify_summary:
            for m in range(int(min(args.num_img_summary, bs))):

                grid_image = make_grid(
                    bbox[m * args.num_cell_h * args.num_cell_w:(m + 1) * args.num_cell_h * args.num_cell_w], 8,
                    normalize=True, pad_value=1
                )
                writer.add_image(f'{prefix}_disc/1-bbox_{m}_{j}', grid_image, global_step)

                grid_image = make_grid(
                    y_each_cell[m * args.num_cell_h * args.num_cell_w:(m + 1) * args.num_cell_h * args.num_cell_w], 8,
                    normalize=True, pad_value=1
                )
                writer.add_image(f'{prefix}_disc/2-y_each_cell_{m}_{j}', grid_image, global_step)

                grid_image = make_grid(
                    o_each_cell[m * args.num_cell_h * args.num_cell_w:(m + 1) * args.num_cell_h * args.num_cell_w], 8,
                    normalize=True, pad_value=1
                )
                writer.add_image(f'{prefix}_disc/3-o_each_cell_{m}_{j}', grid_image, global_step)

                grid_image = make_grid(
                    alpha_each_cell[m * args.num_cell_h * args.num_cell_w:(m + 1) * args.num_cell_h * args.num_cell_w], 8,
                    normalize=True, pad_value=1
                )
                writer.add_image(f'{prefix}_disc/4-alpha_hat_each_cell_{m}_{j}', grid_image, global_step)

                if log_prop_list[j]:
                    bbox_prop = visualize(imgs[m, j].cpu(),
                                          log_prop['z_pres'][m].cpu().detach(),
                                          log_prop['z_where_scale'][m].cpu().detach(),
                                          log_prop['z_where_shift'][m].cpu().detach())

                    grid_image = make_grid(bbox_prop, 5, normalize=True, pad_value=1)
                    writer.add_image(f'{prefix}_prop/1-bbox_{m}_{j}', grid_image, global_step)

                    y_each_obj = log_prop['y_each_obj'][m].view(-1, 3, img_h, img_w).cpu().detach()
                    grid_image = make_grid(y_each_obj, 5, normalize=True, pad_value=1)
                    writer.add_image(f'{prefix}_prop/2-y_each_obj_{m}_{j}', grid_image, global_step)

                    o_each_obj = log_prop['o_each_obj'][m].view(-1, 3, img_h, img_w).cpu().detach()
                    grid_image = make_grid(o_each_obj, 5, normalize=True, pad_value=1)
                    writer.add_image(f'{prefix}_prop/3-o_each_obj_{m}_{j}', grid_image, global_step)

                    alpha_each_obj = log_prop['alpha_hat_each_obj'][m].view(-1, 1, img_h, img_w).cpu().detach()
                    grid_image = make_grid(alpha_each_obj, 5, normalize=True, pad_value=1)
                    writer.add_image(f'{prefix}_prop/4-alpha_each_obj_{m}_{j}', grid_image, global_step)

        bbox_disc = visualize(imgs[:args.num_img_summary, j].cpu(),
                              log_disc['z_pres'][:args.num_img_summary].cpu().detach(),
                              log_disc['z_where_scale'][:args.num_img_summary].cpu().detach(),
                              log_disc['z_where_shift'][:args.num_img_summary].cpu().detach(), only_bbox=True)
        bbox_disc = bbox_disc.view(args.num_img_summary, -1, 3, img_h, img_w)
        bbox_disc = (bbox_disc.sum(dim=1) + imgs[:args.num_img_summary, j].cpu()).clamp(0, 1)
        bbox_disc_list.append(bbox_disc)

    recon_disc = torch.stack(recon_disc_list, dim=1)
    grid_image = make_grid(recon_disc.view(-1, 3, img_h, img_w), seq_len, normalize=True, pad_value=1)
    writer.add_image(f'{prefix}_scalor/3-reconstruction_disc', grid_image, global_step)

    recon_prop = torch.stack(recon_prop_list, dim=1)
    grid_image = make_grid(recon_prop.view(-1, 3, img_h, img_w), seq_len, normalize=True, pad_value=1)
    writer.add_image(f'{prefix}_scalor/4-reconstruction_prop', grid_image, global_step)

    bbox_disc_all = torch.stack(bbox_disc_list, dim=1)
    grid_image = make_grid(bbox_disc_all.view(-1, 3, img_h, img_w),
                           seq_len, normalize=True, pad_value=1)
    writer.add_image(f'{prefix}_scalor/5-bbox_disc', grid_image, global_step)

    bbox_prop_all = torch.stack(bbox_prop_list, dim=1)
    grid_image = make_grid(bbox_prop_all.view(-1, 3, img_h, img_w),
                           seq_len, normalize=True, pad_value=1)
    writer.add_image(f'{prefix}_scalor/6-bbox_prop', grid_image, global_step)

    bg = torch.stack(bg_list, dim=1)
    grid_image = make_grid(bg.view(-1, 3, img_h, img_w), seq_len, normalize=True, pad_value=1)
    writer.add_image(f'{prefix}_scalor/7-background', grid_image, global_step)

    alpha_map = torch.stack(alpha_map_list, dim=1)
    grid_image = make_grid(alpha_map.view(-1, 1, img_h, img_w), seq_len, normalize=False, pad_value=1)
    writer.add_image(f'{prefix}_scalor/8-alpha-map', grid_image, global_step)

    x_mask_color = torch.stack(x_mask_color_list, dim=1)
    grid_image = make_grid(x_mask_color.view(-1, 3, img_h, img_w), seq_len, normalize=False, pad_value=1)
    writer.add_image(f'{prefix}_scalor/9-x-mask-color', grid_image, global_step)

    return

