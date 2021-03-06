import torch
import torch.nn.functional as F
from common import *
import os
import os
import cv2
import numpy as np
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line


def lineplot(xs, ys_population, title, path='', xaxis='episode'):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'
  if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
    ys = np.asarray(ys_population, dtype=np.float32)
    ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
    trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
    data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
  else:
    data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
  plotly.offline.plot({
    'data': data,
    'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)


def write_video(frames, title, path=''):
  frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
  _, H, W, _ = frames.shape
  writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
  for frame in frames:
    writer.write(frame)
  writer.release()

rbox = torch.zeros(3, 21, 21)
rbox[0, :2, :] = 1
rbox[0, -2:, :] = 1
rbox[0, :, :2] = 1
rbox[0, :, -2:] = 1
rbox = rbox.view(1, 3, 21, 21)

gbox = torch.zeros(3, 21, 21)
gbox[1, :2, :] = 1
gbox[1, -2:, :] = 1
gbox[1, :, :2] = 1
gbox[1, :, -2:] = 1
gbox = gbox.view(1, 3, 21, 21)


# color_t = np.random.rand(1000, 3)

def visualize(x, z_pres, z_where_scale, z_where_shift, rbox=rbox, gbox=gbox, only_bbox=False):
    """
        x: (bs, 3, img_h, img_w)
        z_pres: (bs, 4, 4, 1)
        z_where_scale: (bs, 4, 4, 2)
        z_where_shift: (bs, 4, 4, 2)
    """
    bs = z_pres.size(0)
    z_pres = z_pres.view(-1, 1, 1, 1)
    num_obj = z_pres.size(0) // bs
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    bbox = spatial_transform(z_pres * gbox + (1 - z_pres) * rbox,
                             torch.cat((z_scale, z_shift), dim=1),
                             torch.Size([bs * num_obj, 3, img_h, img_w]),
                             inverse=True)
    if not only_bbox:
        bbox = (bbox + torch.stack(num_obj * (x,), dim=1).view(-1, 3, img_h, img_w)).clamp(0.0, 1.0)
    return bbox


def print_scalor(global_step, epoch, local_count, count_inter,
                   num_train, total_loss, log_like, z_what_kl_loss, z_where_kl_loss,
                   z_pres_kl_loss, z_depth_kl_loss, time_inter):
    print(f'Step: {global_step:>5} Train Epoch: {epoch:>3} [{local_count:>4}/{num_train:>4} '
          f'({100. * local_count / num_train:3.1f}%)]    '
          f'total_loss: {total_loss.item():.4f} log_like: {log_like.item():.4f} '
          f'What KL: {z_what_kl_loss.item():.4f} Where KL: {z_where_kl_loss.item():.4f} '
          f'Pres KL: {z_pres_kl_loss.item():.4f} Depth KL: {z_depth_kl_loss.item():.4f} '
          f'[{time_inter:.1f}s / {count_inter:>4} data]')
    return


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count,
              batch_size, num_train):
    # usually this happens only on the start of a epoch
    '''
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'batch_size': batch_size,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'num_train': num_train
    }
    '''
    # let's save the entire model
    ckpt_model_filename = f"ckpt_epoch_{global_step}.pth"
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(model, path)
    print(f'{path:>2} has been successfully saved, global_step={global_step}')
    return


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        checkpoint = torch.load(model_file, map_location=device)

        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            print('loading part of model since key check failed')
            model_dict = {}
            state_dict = model.state_dict()
            for k, v in checkpoint['state_dict'].items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            model.load_state_dict(state_dict)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))

        return step, epoch


def linear_annealing(step, start_step, end_step, start_value, end_value, device):
    if start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step)).to(device)
    elif step >= end_step:
        x = torch.tensor(end_value).to(device)
    else:
        x = torch.tensor(start_value).to(device)

    return x


def spatial_transform(image, z_where, out_dims, inverse=False):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = torch.zeros(2, 3).repeat(image.shape[0], 1, 1).to(image.device)
    # set scaling
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-9)

    # set translation
    theta[:, 0, -1] = z_where[:, 2] if not inverse else - z_where[:, 2] / (z_where[:, 0] + 1e-9)
    theta[:, 1, -1] = z_where[:, 3] if not inverse else - z_where[:, 3] / (z_where[:, 1] + 1e-9)
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims))
    # 3. sample image from grid
    return F.grid_sample(image, grid)


def calc_kl_z_pres_bernoulli(z_pres_logits, prior_pres_prob, eps=1e-15):
    z_pres_probs = torch.sigmoid(z_pres_logits).view(-1)
    prior_pres_prob = prior_pres_prob.view(-1)
    kl = z_pres_probs * (torch.log(z_pres_probs + eps) - torch.log(prior_pres_prob + eps)) + \
         (1 - z_pres_probs) * (torch.log(1 - z_pres_probs + eps) - torch.log(1 - prior_pres_prob + eps))

    return kl

def calc_kl_z_edge_bernoulli(z_edge_logits, prior_edge_prob, eps=1e-15):
    z_edge_probs = F.softmax(z_edge_logits, dim=1)[:,1].view(-1)
    #z_pres_probs = torch.sigmoid(z_pres_logits).view(-1)
    #prior_edge_prob = prior_pres_prob.view(-1)
    kl = z_edge_probs * (torch.log(z_edge_probs + eps) - torch.log(prior_edge_prob + eps)) + \
         (1 - z_edge_probs) * (torch.log(1 - z_edge_probs + eps) - torch.log(1 - prior_edge_prob + eps))

    return kl

