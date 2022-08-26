import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv("mem_v_perf.csv")
print(df["memory"])

x = df["memory"]
y = df["goals"]


barlist = plt.bar(x, y)
barlist[np.argmax(y)].set_color("g")
barlist[np.argmin(y)].set_color("r")
plt.xlabel("Memory size of Deep Q Network")
plt.ylabel("Amount of times goal reached")
plt.title("Bar Graph showing the effect of memory size on agent performance")
plt.show()