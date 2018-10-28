import numpy as np
import gym

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
            # print('Total rewards: {}'.format(totalreward))
            # print('Parameters used: {}'.format(parameters))
            break
    return totalreward


def main():
    bestparams = None  
    bestreward = 0  
    for _ in range(10000):  
        parameters = np.random.rand(4)
        reward = run_episode(env,parameters)
        if reward > bestreward:
            print('Parameters updated at {} round.'.format(_+1))
            print('We managed to last for {} timesteps!'.format(reward))
            bestreward = reward
            bestparams = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 200:
                print(bestparams)
                break


if __name__ == '__main__':
    main()
