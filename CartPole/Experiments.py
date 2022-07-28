from Learning import *
from Architectures import *





def mean_per_episode(model, alg, runs, episodes, env):
    x = range(episodes)
    episode_means = {}
    ys = {}
    counts = []
    for i in range(runs):
        
        if model is not None:
            y, count = alg(model)
        else:
            y, count = alg(env, episodes)
        episode_means[i] = y
        counts.append(count)

    for i in range(episodes):
        ys[i] = [vals[i] for vals in episode_means.values()]

    y = [np.mean(vals) for vals in ys.values()]
    assert(len(x) == len(y))

    return x, y, counts

    




   
   
   

