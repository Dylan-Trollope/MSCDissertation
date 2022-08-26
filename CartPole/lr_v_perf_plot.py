import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv("lr_v_perf.csv")

x = df["lr"]
y = df["goals"]

x_pos = range(0, len(y), 1)

barlist = plt.bar(x_pos, y)
barlist[np.argmax(y)].set_color("g")
plt.xlabel("Learning rate of Deep Q Network")
plt.xticks(x_pos, x)
plt.ylabel("Amount of times goal reached")
plt.title("Bar Graph showing the effect of learning rate on agent performance")
plt.show()