import numpy as np 
from scipy.special import erf, gammainc, gamma
import matplotlib.pyplot as plt

def main():
	n_trials = 1000
	axis = list(range(n_trials)) 
	ns = []
	uns = []
	c2s = []
	uc2s = []
	for _ in range(n_trials):
		n = np.random.normal()
		un = 0.5*erf(n/np.sqrt(2)) 
		ns.append(n)
		uns.append(un)

		c2 = np.random.chisquare(df=2)
		uc2 = (gammainc(1, c2/2.0)/gamma(1))
		c2s.append(c2)
		uc2s.append(uc2)


	plt.scatter(axis, uc2s)
	plt.show()

if __name__ == '__main__':
	main()