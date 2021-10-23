import torch
import gym
import gym_game2048



if __name__ == '__main__':
    board_size = 4
    seed = None
    env = gym.make("game2048-v0", board_size=4, seed=None, binary=True, extractor="cnn", penalty=0)
    c = env.get_board()
    print(c)
    print_board(c)
    print("hello")


