import numpy as np 
import torch 
import torch.nn as nn
import gym 
import sys
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
import matplotlib.colors as plc

from Visualisation import render_averages_plot, render_plot_with_hist

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


fig, axs = plt.subplots(2, 1)
axs[0].set_xticks(range(len(pos_range))[0::3], pos_range[0::3], rotation=90)
axs[0].set_yticks(range(len(ang_range))[0::3], ang_range[0::3])
im = axs[0].imshow(grid, interpolation="nearest", cmap=plt.get_cmap("inferno"))

df = pd.read_csv(sys.argv[2])

pos_range = list(filter(lambda x : x < span and x > -span, np.unique(df["pos"])))
ang_range = list(filter(lambda x : x < span and x > -span, np.unique(df["angle"])))

grid = [[0 for _ in pos_range] for __ in ang_range]

for i, pos in enumerate(pos_range):
	seg = df[df["pos"] == pos]
	for j, ang in enumerate(ang_range):
		cell = seg[seg["angle"] == ang][["q_0", "q_1"]]
		cell = np.max(cell, axis=1)
		cell = np.mean(cell)
		grid[j][i] = cell


axs[1].set_xticks(range(len(pos_range))[0::3], pos_range[0::3], rotation=90)
axs[1].set_yticks(range(len(ang_range))[0::3], ang_range[0::3])

im = axs[1].imshow(grid, interpolation="nearest", cmap=plt.get_cmap("inferno"))
axs[0].title.set_text("First 500")
axs[1].title.set_text("Last 500")
plt.xlabel("Position of Cart")
plt.ylabel("Angle of Pole")
fig.suptitle("Q Value after Reward Engineering the Cart Position - First vs Last 500 steps")
fig.colorbar(im, ax=axs)
plt.show()





