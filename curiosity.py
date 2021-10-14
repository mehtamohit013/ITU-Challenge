#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import tqdm

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import copy
# import multiprocessing as mp
from torch.multiprocessing import Pipe

import gym

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# In[2]:


device = torch.device("cpu")


# In[3]:


import caviar_tools
from beamselect_env import BeamSelectionEnv


# In[4]:


reward_type = 'test'    # 'test' or 'train'
epi = [0,0] #[start,end] 
epi_val = [500,500]

# gym_env_train = BeamSelectionEnv(epi,reward_type)

gym_env_ind = list()
for i in range(epi[0],epi[1]+1):
    gym_env_ind.append(BeamSelectionEnv([i,i],reward_type))

gym_env_val = BeamSelectionEnv(epi_val)


# In[5]:


n_steps_epi = list()
n_steps_epi_val = list()
for i in range(epi[0],epi[1]+1):
    n_steps_epi.append(caviar_tools.linecount([i,i]))

for i in range(epi_val[0],epi_val[1]+1):
    n_steps_epi_val.append(caviar_tools.linecount([i,i]))

n_steps = sum(n_steps_epi)
n_steps_val = sum(n_steps_epi_val)


# In[6]:


train_method = 'ICM'
env_id = None #BreakoutNoFrameskip-v4
env_type = 'beamselect'
env = gym_env_ind[0]


# # Hyper Params

# In[7]:


lam = 0.95
num_worker = 1

num_step = int(128)

ppo_eps = float(0.1)
epoch = int(3)
mini_batch = int(8)
BATCH_SIZE = int(num_step * num_worker / mini_batch) #16
learning_rate = float(1e-4)
entropy_coef = float(0.001)
gamma = float(0.99)
eta = float(1)

clip_grad_norm = float(0.5)

pre_obs_norm_step = int(10)#int(10000)

HISTORY_SIZE = 16
STATES_USED = 13


# In[8]:


input_size = [HISTORY_SIZE,STATES_USED]  
output_size = 192 #64*3


# In[9]:


from utils_cur import *
from agents_cur import *


# In[10]:


reward_rms = RunningMeanStd()
obs_rms = RunningMeanStd(shape=(1, HISTORY_SIZE, 1, STATES_USED))


discounted_reward = RewardForwardFilter(gamma)

agent = ICMAgent


# In[11]:


agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=BATCH_SIZE,
        ppo_eps=ppo_eps,
        eta=eta,
        use_cuda=False,
        use_gae=False,
        use_noisy_net=False
    )


# In[12]:


states = np.zeros([1, HISTORY_SIZE, 1,STATES_USED])

sample_episode = 0
sample_rall = 0
sample_step = 0
sample_env_idx = 0
sample_i_rall = 0
global_update = 0
global_step = 0


# In[13]:


Transition = namedtuple('Transition',
                        ('state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size:int=BATCH_SIZE):
        return random.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)


# In[14]:


history = ReplayMemory(HISTORY_SIZE)

for i in range(HISTORY_SIZE):
    history.push(np.zeros((STATES_USED, )))


# In[15]:


a = history.sample()
np.array(a).shape


# In[16]:


def run(action):
    s, reward, done, info = env.step([action//64, action%64])
    # print(type(s), s.shape)
    history.push(s.astype(np.float))
    
    return [np.array(history.sample(BATCH_SIZE)), reward, done, done, reward]


# In[17]:


# normalize obs
print('Start to initailize observation normalization parameter.....')
next_obs = []
steps = 0
while steps < pre_obs_norm_step:
    steps += num_worker
    actions = np.random.randint(0, output_size, size=(num_worker,))

    for action in actions:
        s, r, d, rd, lr = run(action)
        next_obs.append(s[:])
        
next_obs = np.stack(next_obs)
obs_rms.update(next_obs)
print('End to initalize...')


# In[18]:


while True:
    total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_policy =         [], [], [], [], [], [], [], [], []
    global_step += (num_worker * num_step)
    global_update += 1

    # Step 1. n-step rollout
    for _ in range(num_step):
        actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var)) #Normalization

        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        for action in actions:
            s, r, d, rd, lr = run(action)
            next_states.append(s)
            rewards.append(r)
            dones.append(d)
            real_dones.append(rd)
            log_rewards.append(lr)

        next_states = np.stack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)
        real_dones = np.hstack(real_dones)

        # total reward = int reward
        intrinsic_reward = agent.compute_intrinsic_reward(
            (states - obs_rms.mean) / np.sqrt(obs_rms.var),
            (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
            actions)
        sample_i_rall += intrinsic_reward[sample_env_idx]

        total_int_reward.append(intrinsic_reward)
        total_state.append(states)
        total_next_state.append(next_states)
        total_reward.append(rewards)
        total_done.append(dones)
        total_action.append(actions)
        total_values.append(value)
        total_policy.append(policy)

        states = next_states[:, :, :, :]

        sample_rall += log_rewards[sample_env_idx]

        sample_step += 1
        if real_dones[sample_env_idx]:
            sample_episode += 1
            # writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
            # writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
            # writer.add_scalar('data/step', sample_step, sample_episode)
            sample_rall = 0
            sample_step = 0
            sample_i_rall = 0

    # calculate last next value
    _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))
    total_values.append(value)
    # --------------------------------------------------

    total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, HISTORY_SIZE, 1, STATES_USED])
    total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, HISTORY_SIZE, 1, STATES_USED])
    total_action = np.stack(total_action).transpose().reshape([-1])
    total_done = np.stack(total_done).transpose()
    total_values = np.stack(total_values).transpose()
    total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

    # Step 2. calculate intrinsic reward
    # running mean intrinsic reward
    total_int_reward = np.stack(total_int_reward).transpose()
    total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                        total_int_reward.T])
    mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
    reward_rms.update_from_moments(mean, std ** 2, count)

    # normalize intrinsic reward
    total_int_reward /= np.sqrt(reward_rms.var)
    # writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
    # writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
    # -------------------------------------------------------------------------------------------

    # logging Max action probability
    # writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

    # Step 3. make target and advantage
    target, adv = make_train_data(total_int_reward,
                                    np.zeros_like(total_int_reward),
                                    total_values,
                                    gamma,
                                    num_step,
                                    num_worker)

    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
    # -----------------------------------------------

    # Step 5. Training!
    agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                        (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                        target, total_action,
                        adv,
                        total_policy)

    if global_step % (num_worker * num_step * 100) == 0:
        print('Now Global Step :{}'.format(global_step))
        # torch.save(agent.model.state_dict(), model_path)
        # torch.save(agent.icm.state_dict(), icm_path)


# In[ ]:




