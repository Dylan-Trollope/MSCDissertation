import matplotlib.pyplot as plt 
import numpy as np


def render_plot_with_hist(x, y, count, title, trend, filename):

    f, ax = plt.subplots(nrows=1, ncols=2)
    f.suptitle(title)

    # first of two plots 

    ax[0].plot(y, label="Reward per episode")
    ax[0].axhline(200, label="goal", ls="--", c='red')
    ax[0].set_xlabel("Episode Number")
    ax[0].set_ylabel("Reward")
    ax[0].text(20, 0, "Achieved Goal: " + str(count))
    ax[0].legend()

    if trend: 
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), label="Trend")
        
    # second plot

    ax[1].hist(y[-50:])
    ax[1].axvline(200, label="goal", ls="--", c='red')
    ax[1].set_xlabel("Scores per last 50 episodes")
    ax[1].set_ylabel("Frequency")
    ax[1].legend(
    plt.savefig(fname=filename, dpi=100)
    f.tight_layout()
    plt.show()


def render_averages_plot(x, ys, title, filename):
    plt.title(title)
    plt.axhline(y=200, color='r', linestyle='--', label='goal')
    plt.title("CartPole rewards with no ER over 10 runs")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    for i in range(len(ys)):
        plt.plot(x, ys[i], label='_nolegend_')
    plt.legend()
    plt.savefig(fname=filename, dpi=300)
    plt.show()

