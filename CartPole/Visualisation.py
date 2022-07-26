import matplotlib.pyplot as plt 
import numpy as np


def render_plot(x, y, count, title, trend):

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

	ax[1].hist(y[-50:])
	ax[1].axvline(200, label="goal", ls="--", c='red')
	ax[1].set_xlabel("Scores per last 50 episodes")
	ax[1].set_ylabel("Frequency")
	ax[1].legend()
	
	plt.show()



