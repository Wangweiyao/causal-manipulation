import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import Env
from memory import ExperienceReplay
from utils import lineplot, write_video

import pickle

import cv2

#####################################################################
# Hyperparameters
parser = argparse.ArgumentParser(description='Data')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--trainsteps', type=int, default=300000, metavar='C', help='Collect interval')
parser.add_argument('--save-name', type=str, default='experience', help='name to save')

### Env Specific
parser.add_argument('--env', type=str, default='Pendulum-v0', help='Gym/Control Suite environment')
parser.add_argument('--max-episode-length', type=int, default=100, metavar='T', help='Max episode length')
parser.add_argument('--action-repeat', type=int, default=3, metavar='R', help='Action repeat')
parser.add_argument('--view-nbr', type=int, default=5, metavar='V', help='Number of extra view')

### Memory
parser.add_argument('--experience-size', type=int, default=20000, metavar='D', help='Experience replay size')
parser.add_argument('--seed-episodes', type=int, default=1000, metavar='S', help='Seed episodes')

args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

  
# Setup Logs
results_dir = os.path.join('data', args.id)
os.makedirs(results_dir, exist_ok=True)
logdir =  os.path.join(results_dir, "logs")
os.makedirs(logdir, exist_ok=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
  device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else: 
  device = torch.device('cpu')

# Initialise training environment and experience replay memory
import gap_env
env = Env(args.env, args.view_nbr, args.seed, args.max_episode_length, args.action_repeat)
D = ExperienceReplay(args.experience_size, args.view_nbr, env.observation_size, env.action_size, device)
# Initialise dataset D with S random seed episodes
for s in range(1, args.seed_episodes + 1):
  print(f'Episode: {s}')
  obs, obs2, cam_id = env.reset(); done, t = False, 0
  epdata = []
  while not done:
    action = env.sample_random_action()
    action[2:]=0
    next_obs, next_obs2, next_cam_id, reward, done = env.step(action)
    D.append(obs, obs2, cam_id, action, reward, done)
    epdata.append(next_obs2[0:1])
    obs = next_obs; obs2 = next_obs2; cam_id = next_cam_id
    t += 1

  if s % 100 == 1:
    epdata = np.concatenate(epdata)
    frames = torch.FloatTensor(epdata[:,:3,:,:]) / 255.
    write_video(frames, "Episode"+str(s), logdir)
    print(epdata.shape)
    print(D.observations2.shape)
    goal = np.swapaxes(np.swapaxes(obs[3:], 0, 1), 1, 2) 
    cv2.imwrite(logdir+"/goal"+str(s)+".jpg", cv2.cvtColor(goal, cv2.COLOR_BGR2RGB))

    torch.save(D, os.path.join(results_dir, f'{args.save_name}.pth'),pickle_protocol=4)
    env.close()