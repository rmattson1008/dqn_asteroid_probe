import numpy as np
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle

from my_utils import Transition, ReplayMemory
from models import  DQN_H2, DQN_H5

from tqdm import tqdm as progress_bar



from PIL import Image
from tqdm import tqdm as progress_bar
import cv2
from gym.wrappers import RescaleAction, AtariPreprocessing




#  /home/rachel/temp/dqn_asteroid_probe/training_plots/.png

# EXPERIMENT_NAME =  'asteroids_1000_epoch_h2_tau_.005'
# EXPERIMENT_NAME =  'asteroids-v4_1000_epoch_H2_tau_100'
EXPERIMENT_NAME =  '2_asteroids-v4_1000_epoch_H2_tau_1000'
EXPERIMENT_NAME =  'asteroids-v4_100_epoch_H5_tau_.005'
# t = f"/home/rachel/temp/dqn_asteroid_probe/target_model_test1_500_epoch_h2_checkpoint.pth"
# t_checkpoint = torch.load(t)
print(EXPERIMENT_NAME)
t_checkpoint = torch.load(f'/home/rachel/temp/dqn_asteroid_probe/trained_models/target_model_{EXPERIMENT_NAME}_checkpoint.pth')
# p_checkpoint = torch.load(f'/home/rachel/temp/dqn_asteroid_probe/trained_models/policy_model_{EXPERIMENT_NAME}_checkpoint.pth')
target_net = t_checkpoint['model']
# policy_net = p_checkpoint['model']
target_net.load_state_dict(t_checkpoint['state_dict'])
# policy_net.load_state_dict(p_checkpoint['state_dict'])

print("GETTING TEST activations")
target_net.turn_on_hooks()

env = gym.make("Asteroids-v4", difficulty=0,  render_mode='rgb_array', repeat_action_probability=.25)
# env = AtariPreprocessing(env, frame_skip=1)


seeds = [42,43,44,45,46,47]
for seed in seeds:
    path_to_images = f'/home/rachel/temp/dqn_asteroid_probe/temp_frames/{EXPERIMENT_NAME}_{seed}_gameplay'
    os.makedirs(path_to_images, exist_ok=True)
    observation, info = env.reset()
    images = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_net.to(device)
    actions = []
    activations = []
    images = []
    with torch.no_grad():
        for i_episode in progress_bar(range(1000)):
            # print("1")
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # print("2")
            out, hook_dict = target_net(state)
            action = out.max(1)[1].view(1, 1)
            hook_dict['real_choice'] = action.item()
            activations.append(hook_dict)
            print(action)
           

            # env_screen = env.render()
            # print(env_screen)
            print(state.shape)
            # state = state.permute([0,3,1,2])
            cv2.imwrite(f"{path_to_images}/image{i_episode:03d}.png", state[0].cpu().numpy())
            # img = Image.fromarray(, "RGB")
            # images.append(img)
            # width, height = img.size
            # print(width,height)
            # img.save(f"{path_to_images}/image{i_episode:03d}.png")




            observation, reward, terminated, truncated, info = env.step(action.item())
            # print("4")
            # break
            if terminated or truncated:
                observation, info = env.reset()
                print("Ending Game")
                break
            
            # action = select_action(state)
            # observation, reward, terminated, truncated, _ = env.step(action.item())
            # reward = torch.tensor([reward], device=device)
            # done = terminated or truncated
    # exit()
    print("Out Of Loop")


    # activations = target_net.get_activations()
    # print(actions)
    # for i, d in enumerate(activations):
    #     print(actions[i])
    #     d["real_choice"] = actions[i]
    print(f" activations {type(activations)}")
    # print(activations)
    print(f"number of activations {len(activations)} with keys {activations[0].keys()}")


    print("DICTS")
    print(activations[0]['h1'] == activations[1]['h1'])

    with open(f'/home/rachel/temp/dqn_asteroid_probe/embeddings/{EXPERIMENT_NAME}_{seed}_test_activations.pkl', 'wb') as f:
        pickle.dump(activations, f)

print("exiting")