import gym


""" Q Learning """
# Q-Learning with neural nets
# http://outlace.com/Reinforcement-Learning-Part-3/

""" OPEN AI - Getting started"""
# https://gym.openai.com/docs

""" Running a Environment """

# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action

""" Observations """

# Step function lets us see the effect of actions in an environment.
# It return 4 values:
#   observation (object): environment-specific-object, representing observation of the environment
#   reward (float): amount of reward achieved by the previous action
#   done (boolean): whether it's time to reset the environment again
#   info (dict): diagnostic information useful for debugging


env = gym.make('FrozenLake-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(int(raw_input("-->")))
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break

""" Spaces """


