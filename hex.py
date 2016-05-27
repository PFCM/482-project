"""do the UCT on hex maybe"""
import time
import gym


def main():
    env = gym.make('Hex9x9-v0')
    observation = env.reset()

    for t in range(100):
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        time.sleep(10)
        if done:
            break


if __name__ == '__main__':
    main()
