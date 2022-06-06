import argparse
import os
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
from log_utils import log_summary
from utils import save_ckpt, load_ckpt, print_scalor
from common import *
import parse

from tensorboardX import SummaryWriter
from memory import ExperienceReplay
from scalor import SCALOR

def main(args):
    args.color_t = torch.rand(700, 3)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)

    device = torch.device(
       "cuda" if not args.nocuda and torch.cuda.is_available() else "cpu")

    model = SCALOR(args)


    if args.last_ckpt:
        print(f"Load ckpt: {args.last_ckpt}")
        model = torch.load(args.last_ckpt, map_location=device)
        #global_step, args.start_epoch = \
        #    load_ckpt(model, optimizer, args.last_ckpt, device)
    
    model.to(device)
    model.train()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    global_step = 0

    writer = SummaryWriter(args.summary_dir)

    args.global_step = global_step

    log_tau_gamma = np.log(args.tau_end) / args.tau_ep


    D = torch.load(args.experience_replay)
    num_train = D.size

    for epoch in range(int(args.start_epoch), args.epochs):
        local_count = 0
        last_count = 0
        end_time = time.time()


        for _ in range(num_train // args.batch_size):

            chunk_size = 3 + epoch #// 10
            chunk_size = min(chunk_size, args.chunk_size)
            #chunk_size = 2#args.chunk_size

            observations, actions, rewards, nonterminals = D.sample(args.batch_size, chunk_size)
            
            tau = np.exp(global_step * log_tau_gamma)
            tau = max(tau, args.tau_end)
            args.tau = tau

            global_step += 1

            log_phase = global_step % args.print_freq == 0 or global_step == 1
            args.global_step = global_step
            args.log_phase = log_phase

            if np.random.binomial(1, min(epoch, 10)/10, 1)[0] and not log_phase:
                args.phase_generate = True
            else:
                args.phase_generate = False

            sample = observations[:,:,0:3].permute(1,0,2,3,4) / 255
            actions = actions.permute(1,0,2)

            imgs = sample.to(device)
            actions = actions.to(device)

            y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
            kl_z_pres, kl_z_bg, kl_edge_type, counting, \
            log_disc_list, log_prop_list, scalor_log_list = model(imgs, actions)

            log_like = log_like.mean(dim=0)
            kl_z_what = kl_z_what.mean(dim=0)
            kl_z_where = kl_z_where.mean(dim=0)
            kl_z_depth = kl_z_depth.mean(dim=0)
            kl_z_pres = kl_z_pres.mean(dim=0)
            kl_z_bg = kl_z_bg.mean(0)
            kl_edge_type = kl_edge_type.mean(0)

            total_loss = - log_like + kl_edge_type + kl_z_pres + kl_z_where + kl_z_depth + kl_z_bg + kl_z_what

            optimizer.zero_grad()
            total_loss.backward()

            clip_grad_norm_(model.parameters(), args.cp)
            optimizer.step()

            local_count += imgs.data.shape[0]

            if log_phase:

                time_inter = time.time() - end_time
                end_time = time.time()

                count_inter = local_count - last_count

                print_scalor(global_step, epoch, local_count, count_inter,\
                               num_train, total_loss, log_like, kl_z_what, kl_z_where,\
                               kl_z_pres, kl_z_depth, time_inter)

                writer.add_scalar('train/total_loss', total_loss.item(), global_step=global_step)
                writer.add_scalar('train/log_like', log_like.item(), global_step=global_step)
                writer.add_scalar('train/What_KL', kl_z_what.item(), global_step=global_step)
                writer.add_scalar('train/Where_KL', kl_z_where.item(), global_step=global_step)
                writer.add_scalar('train/Pres_KL', kl_z_pres.item(), global_step=global_step)
                writer.add_scalar('train/Depth_KL', kl_z_depth.item(), global_step=global_step)
                writer.add_scalar('train/Bg_KL', kl_z_bg.item(), global_step=global_step)
                writer.add_scalar('train/Edge_KL', kl_edge_type.item(), global_step=global_step)
                writer.add_scalar('train/tau', tau, global_step=global_step)

                log_summary(args, writer, imgs, y_seq, global_step, log_disc_list,
                            log_prop_list, scalor_log_list, prefix='train')

                last_count = local_count

                #print(args.generate_freq)
                #args.generate_freq = 2
                #if global_step % args.generate_freq == 0:
                ####################################### do generation ####################################
                model.eval()
                with torch.no_grad():
                    args.phase_generate = True
                    y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                    kl_z_pres, kl_z_bg, kl_edge_type, counting, \
                    log_disc_list, log_prop_list, scalor_log_list = model(imgs, actions)
                    args.phase_generate = False
                    
                    log_summary(args, writer, imgs, y_seq, global_step, log_disc_list,
                                log_prop_list, scalor_log_list, prefix='generate')
                model.train()
                ####################################### end generation ####################################

            if global_step % args.save_epoch_freq == 0 or global_step == 1:
                save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                          local_count, args.batch_size, num_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCALOR')
    args = parse.parse(parser)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)

