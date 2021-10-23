import gym
from stable_baselines3.common.env_checker import check_env
from actor_critic import Actor,Critic
import torch.nn as nn
import  torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym_game2048
from numba import uint32, int32, int32, b1, float64, float32
torch.autograd.set_detect_anomaly(True)
from itertools import count



env = gym.make("game2048-v0", board_size=4, seed=None, binary=True, extractor="mlp", penalty=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = 4
print("Observation Size:", STATE_SIZE, "Action Size:", ACTION_SIZE)
LEARNING_RATE = 0.000001
GAMMA = 0.999
NUM_ITERATIONS = 10000

actor = Actor(STATE_SIZE, ACTION_SIZE).to(device)
critic = Critic(STATE_SIZE, ACTION_SIZE).to(device)

optimizerA = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
optimizerC = optim.Adam(critic.parameters(), lr=LEARNING_RATE)


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def print_board(board):
    print("-------------")
    for line in board:
        print("+",*line,"+", sep = " ")

    print("-------------")



def test_env():
    pass


def train_iterations():
    for iter in range(NUM_ITERATIONS):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        env.reset()
        total_reward = 0
        flag = False
        while not flag:
            env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            total_reward += reward
            flag = done
            log_prob = dist.log_prob(action).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration', iter, " Reward:", total_reward)
                print_board(env.get_board())
                break

        next_state = torch.FloatTensor(next_state).to(dxevice)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    train_iterations()
    print("g")
