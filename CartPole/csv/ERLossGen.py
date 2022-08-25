import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
    
df = pd.read_csv("loss_graph.csv")
# print(df)

x = len(df)
y = df["loss_vals"].to_list()


plt.scatter(range(len(y)), y)

z = np.polyfit(range((len(y))), y, 2)
p = np.poly1d(z)
plt.plot(range(len(y)), p(range(len(y))), label="Trend", ls="--", color="red")
plt.xlabel("Episodes")
plt.ylabel("Average Loss per Episode")
plt.title("Loss over time for DQN with Experience Replay")
plt.show()
