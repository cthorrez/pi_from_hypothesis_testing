import numpy as np
import matplotlib.pyplot as plt


def main():
	trials = 1000
	xs = []
	ys = []
	for t in range(trials):
		x = np.random.normal()
		y = np.random.normal()
		xs.append(x)
		ys.append(y)
	plt.scatter(xs,ys, s=1)
	circle = plt.Circle((0, 0), 1, color='r', fill=False, lw=3)
	square = plt.Rectangle([-1,-1],2,2, color='r', fill=False, lw=3)
	plt.gcf().gca().add_artist(circle)
	plt.gcf().gca().add_artist(square)
	plt.axis([-2.5,2.5,-2.5,2.5])
	plt.title('Bivariate Gaussian with Acceptance Regions')
	fig = plt.gcf()
	fig.set_size_inches(5, 5)
	plt.savefig('figures/bivariate_gaussian.png')

if __name__ == '__main__':
	main()