#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import tqdm
import pickle
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

gym_env_train = BeamSelectionEnv(epi,reward_type)

# gym_env_ind = list()
# for i in range(epi[0],epi[1]+1):
#     gym_env_ind.append(BeamSelectionEnv([i,i],reward_type))

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


n_steps_val


# In[7]:


train_method = 'ICM'
env_id = None #BreakoutNoFrameskip-v4
env_type = 'beamselect'
env = gym_env_train


# # Hyper Params

# In[8]:


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


# In[9]:


model_path = './model_curiosity'
icm_path = './icm_curiosity'


# In[10]:


input_size = [HISTORY_SIZE,STATES_USED]  
output_size = 192 #64*3


# In[11]:


from utils_cur import *
from agents_cur import *


# In[12]:


reward_rms = RunningMeanStd()
obs_rms = RunningMeanStd(shape=(1, HISTORY_SIZE, 1, STATES_USED))


discounted_reward = RewardForwardFilter(gamma)

agent = ICMAgent


# In[13]:


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


# In[14]:


states = np.zeros([1, HISTORY_SIZE, 1,STATES_USED])

sample_episode = 0
sample_rall = 0
sample_step = 0
sample_env_idx = 0
sample_i_rall = 0
global_update = 0


# In[15]:


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


# In[16]:


history = ReplayMemory(HISTORY_SIZE)

for i in range(HISTORY_SIZE):
    history.push(np.zeros((STATES_USED, )))


# In[17]:


a = history.sample()
np.array(a).shape


# In[18]:


def run(action, env):
    s, reward, done, info = env.step([action//64, action%64])
    # print(type(s), s.shape)
    history.push(s.astype(np.float))
    
    return [np.array(history.sample(BATCH_SIZE)), reward, done, done, reward]


# In[19]:


# normalize obs
print('Start to initailize observation normalization parameter.....')
next_obs = []
steps = 0
while steps < pre_obs_norm_step:
    steps += num_worker
    actions = np.random.randint(0, output_size, size=(num_worker,))

    for action in actions:
        s, r, d, rd, lr = run(action, gym_env_train)
        next_obs.append(s[:])
        
next_obs = np.stack(next_obs)
obs_rms.update(next_obs)
print('End to initalize...')


# In[20]:


f = open('obs_rms.pkl', 'wb')
pickle.dump(obs_rms, f)
f.close()


# In[21]:


def val(env):
    with torch.no_grad():
        history = ReplayMemory(HISTORY_SIZE)
        for i in range(HISTORY_SIZE):
            history.push(np.zeros((STATES_USED, )))
        f = open('obs_rms.pkl', 'rb')
        obs_rms = pickle.load(f)
        rall = 0
        rd = False
        intrinsic_reward_list = []
        states = np.zeros([1, HISTORY_SIZE, 1,STATES_USED])
        def run(action):
            s, reward, done, info = env.step([action//64, action%64])
            # print(type(s), s.shape)
            history.push(s.astype(np.float))

            return [np.array(history.sample(BATCH_SIZE)), reward, done, done, reward]

        for steps in range(n_steps_val):
            actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []

            for action in actions:
                s, r, d, rd, lr = run(action)
                rall += r
                next_states.append(s)
        #         next_obs = s[3, :, :].reshape([1, 1, 1,STATES_USED])

            # total reward = int reward + ext Reward
        #     intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
        #     intrinsic_reward_list.append(intrinsic_reward)
            next_states = np.stack(next_states)
            states = next_states[:, :, :, :]

        #     if rd:
        #         intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
        #             intrinsic_reward_list)
        #         with open('int_reward', 'wb') as f:
        #             pickle.dump(intrinsic_reward_list, f)
        #         steps = 0
        #         rall = 0
        print(f"Total Val Reward: {rall}, Avg Val Reward: {rall/n_steps_val}")


# In[22]:


running_total_reward = 0
cnt = 0
for global_step in range(0, n_steps+100000000, 128):
    total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_policy =         [], [], [], [], [], [], [], [], []
    global_update += 1

    # Step 1. n-step rollout
    for _ in range(num_step):
        actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var)) #Normalization

        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        
        if (len(actions) == 0):
            print("yesss")
        for action in actions:
            s, r, d, rd, lr = run(action, gym_env_train)
            cnt += 1
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

    running_total_reward += np.sum(total_reward)

    
    if (global_step) % (num_worker * num_step) == 0:
        print('Now Global Step :{}'.format((global_step)))
        print(f'Total reward : {np.mean(total_reward)}, Running: {running_total_reward/(global_step+num_step)} {cnt}')
#         torch.save(agent.model.state_dict(), model_path)
#         torch.save(agent.icm.state_dict(), icm_path)

#     if (global_step % 1280) == 0:
#         val(copy.deepcopy(gym_env_val))

