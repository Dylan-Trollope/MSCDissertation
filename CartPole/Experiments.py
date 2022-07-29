from Learning import *
from Architectures import *





def mean_per_episode(model, alg, episodes, runs):
	x = range(episodes)
	episode_means = {}
	ys = {}
	counts = []
	for i in range(runs):
		y, count = alg(model)
		episode_means[i] = y
		counts.append(count)

	for i in range(episodes):
		ys[i] = [vals[i] for vals in episode_means.values()]

	y = [np.mean(vals) for vals in ys.values()]
	assert(len(x) == len(y))
	print(counts)
	return x, y, counts

	



   
   
   

