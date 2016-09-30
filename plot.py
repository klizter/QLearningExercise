import numpy as np
from matplotlib import pyplot as plt


class Plot:

    def __init__(self):
        pass

    @classmethod
    def plot_evolution(cls, episodes_reward):
        plt.figure(num=None, figsize=(100, 10), dpi=30, facecolor='w', edgecolor='k')
        plt.subplot(221)
        plt.xlim(1, len(episodes_reward))

        generations = np.linspace(1, len(episodes_reward), len(episodes_reward), endpoint=True)

        plt.plot(generations, episodes_reward, linewidth=1.0, color="green", linestyle="solid")

        plt.show()
