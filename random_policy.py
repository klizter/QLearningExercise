import gym


class RandomPolicy:

    def __init__(self, env_name):
        self.env_name = env_name

    def run(self):
        env = gym.make(self.env_name)
        for i_episode in range(20):
            observation = env.reset()
            for t in range(100):
                env.render()
                print(observation)
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                if done:
                    print "Episode finished after {} timesteps".format(t + 1)
                    break


def exercise_one():

    raw_input('Click any button to run random policy for Frozen Lake')
    RandomPolicy('FrozenLake-v0').run()

    raw_input("Click any button to run random policy for Taxi Environment")
    RandomPolicy('Taxi-v1').run()

exercise_one()