import matplotlib.pyplot as plt

goals_runs = [87, 83, 64, 50, 61, 92, 77, 46, 72, 69]
plt.boxplot(goals_runs, meanline=True)
plt.title("Box Plot showing summary of statistics for 10 runs with ER")
plt.ylabel("Amount of times goal reached")
plt.xlabel("")
plt.show()
