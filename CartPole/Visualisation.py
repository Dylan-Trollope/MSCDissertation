import matplotlib.pyplot as plt 
import numpy as np


def render_plot(x, y, title,  x_label, y_label, trend):
	plt.plot(x, y)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	if trend: 
		z = np.polyfit(x, y, 1)
		p = np.poly1d(z)
		plt.plot(x, p(x))
	
	plt.show()



