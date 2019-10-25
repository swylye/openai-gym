import logging
import os

import gym
import numpy as np

from util.logger import log_setup

env = gym.make('CartPole-v0')

# parameters = np.random.rand(4)


def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0.5 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            # logger.info('Total rewards: {}'.format(totalreward))
            # logger.info('Parameters used: {}'.format(parameters))
            break
    return totalreward


def main():
    bestparams = None  
    bestreward = 0  
    for _ in range(10000):  
        parameters = np.random.rand(4)
        reward = run_episode(env,parameters)
        if reward > bestreward:
            logger.info('Parameters updated at {} round.'.format(_+1))
            logger.info('We managed to last for {} timesteps!'.format(reward))
            bestreward = reward
            bestparams = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 200:
                logger.info(bestparams)
                break


if __name__ == '__main__':
    # Global variables
    LOGGER_LEVEL = logging.INFO  # TODO: Set logging level | Options: 'INFO', 'DEBUG', 'ERROR', etc.

    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_NAME = os.path.basename(__file__)[:-3]
    LOG_FILENAME = SCRIPT_NAME
    logger = log_setup(PARENT_DIR, LOG_FILENAME, LOGGER_LEVEL)

    main()