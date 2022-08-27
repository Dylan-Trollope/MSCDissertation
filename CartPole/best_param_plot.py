import numpy as np
import pandas as pd
from Visualisation import *


df = pd.read_csv("best_perf.csv")

x = df["rewards"].to_list()
count = len(df[df["rewards"]>=500])
print(count)
#render_plot_with_hist(range(len(x)), x, count, "Agent Performance in CartPole with best parameters", True)
