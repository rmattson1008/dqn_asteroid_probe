import numpy as np
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from my_utils import Transition, ReplayMemory
from models import  DQN_H2, DQN_H5, DQN_FLAT

from tqdm import tqdm as progress_bar

# TODO - need to be more familiar w atari gym. 
# TODO - find difficulty setting. 
# obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        with open('/home/rachel/temp/dqn_asteroid_probe/pytorch_src/actions.txt', 'a') as f:
            f.write("R")
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

    plt.savefig(f'/home/rachel/temp/dqn_asteroid_probe/training_plots/duration_{EXPERIMENT_NAME}.png')
    plt.close() 
    return



    #TODO - write to file or tensorboard. 


def optimize_model():

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    k_loss = loss.cpu()
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return k_loss


def plot_rewards(total_rewards):

    plt.figure(1)
    plt.title('Training Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.plot(total_rewards)
    plt.savefig(f'/home/rachel/temp/dqn_asteroid_probe/training_plots/rewards_{EXPERIMENT_NAME}.png')
    plt.close() 

    # plt.figure(1)
    # plt.title('Training Rewards (mean)')
    # plt.xlabel('Step')
    # plt.ylabel('Mean Reward')
    # plt.plot(smoothed_rewards)
    # plt.savefig(f'/home/rachel/temp/dqn_asteroid_probe/training_plots/mean_rewards_{EXPERIMENT_NAME}.png')
    # plt.close() 
    return

if __name__ == '__main__':
    env = gym.make("Asteroids-v4", difficulty=0)
    # env = gym.make("ALE/Alien-v5", difficulty=1)

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}")


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = .005
    LR = 1e-4
    # EXPERIMENT_NAME = '2_asteroids-v4_1000_epoch_H2_tau_1000'
    # EXPERIMENT_NAME = 'asteroids-v4_1000_epoch_H2_tau_100'
    # EXPERIMENT_NAME = 'asteroids-v4_100_epoch_H2_tau_.005'
    # EXPERIMENT_NAME = 'asteroids-v4_100_epoch_H5_tau_.005'
    EXPERIMENT_NAME = 'asteroids-v4_100_epoch_FLAT_tau_.005'

    losses = []

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)
    print("NOBSERVATIONS", state.shape)
    # exit()

    # t_checkpoint = torch.load(f'/home/rachel/temp/dqn_asteroid_probe/trained_models/target_model_{EXPERIMENT_NAME}_checkpoint.pth')
    # p_checkpoint = torch.load(f'/home/rachel/temp/dqn_asteroid_probe/trained_models/policy_model_{EXPERIMENT_NAME}_checkpoint.pth')
    # target_net = t_checkpoint['model']
    # policy_net = p_checkpoint['model']
    # target_net.load_state_dict(t_checkpoint['state_dict'])
    # policy_net.load_state_dict(p_checkpoint['state_dict'])

    policy_net = DQN_FLAT(n_observations, n_actions).to(device)
    target_net = DQN_FLAT(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0
    episode_durations = []


    if torch.cuda.is_available():
        num_episodes = 100
    else:
        num_episodes = 50


    total_rewards = []
    # mean_rewards = [] 


    for i_episode in progress_bar(range(num_episodes)):
        #write over a copy of the model every 100 episodes
        if i_episode %100 == 0:
            checkpoint = {'model': target_net, 'state_dict': target_net.state_dict()}
            torch.save(checkpoint, f'/home/rachel/temp/dqn_asteroid_probe/trained_models/target_model_{EXPERIMENT_NAME}_checkpoint.pth')
            checkpoint = {'model': policy_net, 'state_dict': policy_net.state_dict()}
            torch.save(checkpoint, f'/home/rachel/temp/dqn_asteroid_probe/trained_models/policy_model_{EXPERIMENT_NAME}_checkpoint.pth')
        rewards = []
        # t.set_description(f'val{}')
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        e_loss = []
        with open('/home/rachel/temp/dqn_asteroid_probe/pytorch_src/actions.txt', 'a') as f:
                f.write("\n")
        for t in count():
            action = select_action(state)
            with open('/home/rachel/temp/dqn_asteroid_probe/pytorch_src/actions.txt', 'a') as f:
                f.write(str(action.cpu().item()))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model()
            if loss:
                e_loss.append(loss.detach())
            

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                total_rewards.append(np.sum(rewards))
                # mean_rewards.append(np.mean(rewards[-100:]))
                
                episode_durations.append(t + 1)
                m_loss = np.mean(e_loss)
                losses.append(m_loss)
                plot_durations()
                plot_rewards(total_rewards)

                # print(loss)
                # losses.append(loss.numpy())
                plt.figure(1)
                plt.title('Training Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.plot(losses)
                plt.savefig(f'/home/rachel/temp/dqn_asteroid_probe/training_plots/loss_{EXPERIMENT_NAME}.png')
                plt.close() 

                break

    plot_durations(show_result=True)

     
    
    print("Saving checkpoint")
    print(target_net)

    with open(f'{EXPERIMENT_NAME}_total_rewards.pkl', 'wb') as f:
        pickle.dump(total_rewards, f)

    with open(f'{EXPERIMENT_NAME}_durations.pkl', 'wb') as f:
        pickle.dump(episode_durations, f)
    
    # with open(f'{EXPERIMENT_NAME}_mean_rewards.pkl', 'wb') as f:
    #     pickle.dump(mean_rewards, f)

    # I dont know why I get hook error. hmm
    #try popping off hooks... 
    checkpoint = {'model': target_net, 'state_dict': target_net.state_dict()}
    torch.save(checkpoint, f'/home/rachel/temp/dqn_asteroid_probe/trained_models/target_model_{EXPERIMENT_NAME}_checkpoint.pth')
    checkpoint = {'model': policy_net, 'state_dict': policy_net.state_dict()}
    torch.save(checkpoint, f'/home/rachel/temp/dqn_asteroid_probe/trained_models/policy_model_{EXPERIMENT_NAME}_checkpoint.pth')


    ######################

    t_checkpoint = torch.load(f'/home/rachel/temp/dqn_asteroid_probe/trained_models/target_model_{EXPERIMENT_NAME}_checkpoint.pth')
    p_checkpoint = torch.load(f'/home/rachel/temp/dqn_asteroid_probe/trained_models/policy_model_{EXPERIMENT_NAME}_checkpoint.pth')
    target_net = t_checkpoint['model']
    policy_net = p_checkpoint['model']
    target_net.load_state_dict(t_checkpoint['state_dict'])
    policy_net.load_state_dict(p_checkpoint['state_dict'])

    # print("GETTING TEST activations")
    # target_net.turn_on_hooks()

    # with torch.no_grad():
    #     for i_episode in progress_bar(range(1)):
    #         state, info = env.reset()
    #         state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    #         out = target_net(state).max(1)[1].view(1, 1)
    #         print("Out", out)
            
    #         # action = select_action(state)
    #         # observation, reward, terminated, truncated, _ = env.step(action.item())
    #         # reward = torch.tensor([reward], device=device)
    #         # done = terminated or truncated


    # activations = target_net.get_activations()
    # print(f" activations {type(activations)}")
    # # print(activations)
    # print(f"number of activations {len(activations)} with keys {activations[0].keys()}")
    
    # with open(f'{EXPERIMENT_NAME}_test_activations.pkl', 'wb') as f:
    #     pickle.dump(activations, f)


    #TODO - need better plot saving
    # TODO set seeds         
    # TODO -if not working, dial up replay memory
    print('Complete')




