import numpy as np 
import torch 
import torch.nn as nn
import gym 
import sys
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
import matplotlib.colors as plc


df = pd.read_csv(sys.argv[1])
span = 1.0

pos_range = list(filter(lambda x : x < span and x > -span, np.unique(df["pos"])))
ang_range = list(filter(lambda x : x < span and x > -span, np.unique(df["angle"])))

print(pos_range)
print(ang_range)


grid = [[0 for _ in pos_range] for __ in ang_range]

for i, pos in enumerate(pos_range):
	seg = df[df["pos"] == pos]
	for j, ang in enumerate(ang_range):
		cell = seg[seg["angle"] == ang][["q_0", "q_1"]]
		cell = np.max(cell, axis=1)
		cell = np.mean(cell)
		grid[j][i] = cell



plt.xticks(range(len(pos_range))[0::3], pos_range[0::3], rotation=90)
plt.yticks(range(len(ang_range))[0::3], ang_range[0::3])
plt.imshow(grid, cmap=plt.get_cmap("inferno"))
plt.colorbar()
plt.show()
