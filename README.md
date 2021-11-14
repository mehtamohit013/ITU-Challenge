# PS-006-ML5G-PHY-Reinforcement-learning_IITI-RL

>This repository is copy of challenge's main [repo](https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY-RL) and the code is modified by following the challenge's rules and guidelines. 

# Submission

Presentation and Videos (for best return and worst return respectively) are available at following [link](https://drive.google.com/drive/u/0/folders/1bo3ceT3fknYZM8-kKN6Ud5NIPy9K3_dY). 

# Reproducing Results

Curiosity model is trained using the script [curiosity.ipynb](./curiosity.ipynb) and the model weights is saved in the directory [model_state_dict](./model_state_dict).

To reproduce the leaderboard results, please run [curiosity_eval.ipynb](./curiosity_eval.ipynb).

# Baselines

The following are the baselines we tried:

1. DQN: [RL.ipynb](./RL.ipynb) and [RL_fed.ipynb](./RL_fed.ipynb)
2. [Advanced Actor Critic](./A2C.ipynb)
3. [Policy Gradient Network](./policy_grad.ipynb)

# References

1. https://github.com/jcwleo/curiosity-driven-exploration-pytorch
2. https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
3. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
4. https://tims457.medium.com/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
