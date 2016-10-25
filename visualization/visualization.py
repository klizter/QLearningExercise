import gym


def visualize_learning(environment_name, q_learning):

    env = gym.make(environment_name)

    while True:

        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            raw_input("CLICK ENTER FOR NEXT ACTION")
            action = q_learning.greedy_probability_policy(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print "Episode finished after {} timesteps".format(t + 1)
                break

        print "Next episode (1) or exit (2)"
        option = raw_input("-->")

        if option == "1":
            continue
        else:
            break


