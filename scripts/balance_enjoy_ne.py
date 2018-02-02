import random
import time
import pickle

import numpy as np

import balance_bot_ne

def main():
    policy_data = pickle.load(open("best.pkl", "rb"))
    policy = balance_bot_ne.NeuralNet.with_dictionary(policy_data)

    env = balance_bot_ne.BalancebotEnvUneven(vdrange=(-1.0, 1.0), render=True)

    while True:
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs, reward, done = env.step(policy.forward(obs)[0])
            total_reward += reward
            time.sleep(0.005)

if __name__ == "__main__":
    main()