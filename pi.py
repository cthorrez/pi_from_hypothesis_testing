import numpy as np 
from scipy.special import erf, gammainc, gamma
from multiprocessing import Pool
from copy import deepcopy
from collections import deque


def get_dataset(n, Beta, sigma2_e):
	# generate 2 columns of random numbers
	x = np.random.rand(n,2)
	ones = np.ones([n,1])
	x = np.hstack([ones,x])

	# generate targets
	y = np.dot(x,np.array([Beta]).T) 
	
	# sample errors from gaussian
	e = np.random.normal(loc=0, scale=np.sqrt(sigma2_e), size=[n,1])

	# target = tarteg + noise
	t = y + e
	return x,t


def get_ols_estimator(x,t):
	"""Calculates the OLS estimator"""
	return np.dot(np.linalg.inv(np.dot(x.T,x)), np.dot(x.T,t))

def get_z_statistic(x,t,Beta_hat,idx,value):
	"""Calculates the z statistic for the hypothesis: Beta[idx] = value"""
	s = np.sqrt(get_s2(x,t,Beta_hat))
	XTXi = np.linalg.inv(np.dot(x.T,x))
	z = (Beta_hat[idx] - value)/(s*np.sqrt(XTXi[idx,idx]))
	return z

def get_wald_statistic(x,t,Beta_hat,C,b):
	"""Calculates the wald statistic for the hypothesis specified in constraint matrix C. C*Beta = b
	Uses the equation from stats 413 lecture notes"""
	s2 = get_s2(x,t,Beta_hat)
	XTXi = np.linalg.inv(np.dot(x.T,x))
	CXTXiCTi = np.linalg.inv(np.dot(np.dot(C,XTXi),C.T))
	CBmb = np.dot(C,Beta_hat) - b
	w = np.dot(np.dot(CBmb.T,CXTXiCTi),CBmb)/s2
	return w

def get_wald_statistic2(x,t,Beta_hat,C,b):
	"""Also calculates the wald statistic. Using the equation on wikipedia"""
	n = x.shape[0]
	CBmb = np.dot(C,Beta_hat) - b
	avar = get_avar(x,t,Beta_hat)
	CAvarnCT = np.linalg.inv(np.dot(np.dot(C,avar/n),C.T))
	w  = np.dot(np.dot(CBmb.T, CAvarnCT),CBmb)
	return w


def get_avar(x,t,Beta_hat,unbiased=True):
	"""Calculates the asymptotic variance of the OLS estimator"""
	n = x.shape[0]
	pred = np.dot(x,Beta_hat)
	s2 = np.sum(np.power(t-pred,2))/(n-int(unbiased))
	XTXin = np.linalg.inv(np.dot(x.T,x)/n)
	avar = s2*XTXin
	return avar

def get_s2(x,t,Beta_hat,unbiased=True):
	"""Calculates an estimate of variance for an estimator"""
	n = x.shape[0]
	pred = np.dot(x,Beta_hat)
	s2 = np.sum(np.power(t-pred,2))/(n-int(unbiased))
	return s2

def normal2uniform(n, lower, upper):
	"""Converts a sample from a normal distribution to a sample from the uniform 
	distribution over the interval (lower,upper)"""
	mid = (upper + lower)/2
	span = (upper - lower)
	# convert to U(-0.5,0.5) via CDF then scale and shift
	u = 0.5*erf(n/np.sqrt(2))*span + mid
	return u

class ChiSquare2UniformSquare:
	"""Use function objct for chi square to uniform square conversion becasue
	it requires memory of past samples """
	def __init__(self):
		"""cache of previous samples is used to decorrelate gaussian samples over time"""
		self.prevs = deque(maxlen=1)
		# initial entry to cache is sqrt(2)/2 because which is the square root of half of
		# the mean of a chi squred with 2 degrees of freedom
		self.prevs.append(np.sqrt(2)/2.0)
		self.prev = np.random.choice(self.prevs)


	def __call__(self,c,df,lower,upper):
		"""Converts a sample from a chi squred distribution with df degrees of freedom 
		to a sample from a sum of squared uniform distributions over the interval (lower,upper)"""
		mid = (upper + lower)/2
		span = (upper - lower)

		# second method of conversion via th CDF. Converts to U(0,1) then scale and shift
		u = (gammainc(df/2, c/2.0)/gamma(df/2))*span+lower
		
		# square and add the uniforms
		u2 = np.power(u,2) + np.power(self.prev,2)
		
		# store the current sample
		self.prevs.append(u[0,0])
		self.prev = np.random.choice(self.prevs)
		return u2



def do_n_tests_wrapper(kwargs):
	"""wrapper because multiprocessing map cannot send unwrapped kwargs dicts"""
	return do_n_tests(**kwargs)

def do_n_tests(seed, n_t, n_d, Beta, sigma2_e, C, b, chisquare2uniformsquare, 
			   u_lower, u_upper, z_cut, w_cut, p_num):
	np.random.seed(seed)
	w_accept = 0
	z_accept = 0
	# the chi-squared has as many degrees of freedom as there
	# are constraints in the null hypothesis
	df = C.shape[0]

	for i in range(n_t):

		if (p_num == 0) and ((i/float(n_t))*100 %1 == 0):
			print int((i/float(n_t)*100)), '%'

		# constuct data set
		x,t =  get_dataset(n_d, Beta, sigma2_e)

		# fit ordinary least squares regression to the data
		Beta_hat = get_ols_estimator(x,t)

		# compute test statistics
		z_1 = get_z_statistic(x, t, Beta_hat, 1, Beta[1])
		z_2 = get_z_statistic(x, t, Beta_hat, 2, Beta[2])
		w = get_wald_statistic(x, t, Beta_hat, C, b)

		# convert test statistics to uniform
		u_z1 = normal2uniform(z_1, u_lower, u_upper)
		u_z2 = normal2uniform(z_2, u_lower, u_upper)
		u_w = chisquare2uniformsquare(w, df, u_lower, u_upper)

		# check test statistics vs threshods to determine accept or reject
		if (abs(u_z1) < z_cut) and (abs(u_z2) < z_cut):
			z_accept += 1

		if u_w < w_cut:
			w_accept += 1

	pi = 4.0*(w_accept / float(z_accept))
	return pi




def main():
	# initialize linear equation: t = 2*x1 + 5*x2 + 0 (no bias term)
	Beta = np.array([0,1,-1], dtype=float)
	
	# variance of Gaussian noise added to targets of the data set
	sigma2_e = 1.0

	# C and b specify constraints that Beta_hat[1] = Beta[1] and Beta_hat[2] = Beta[2]
	C = np.array([[0,1,0],
		 		 [0,0,1]])
	b = np.array([[Beta[1],Beta[2]]]).T
		
	iters = 1000000				# number of data sets to construct and test
	n_d = 1000					# number of points per data set
	n_processes = 8				# number of processes to split the work across
	n_t = iters/n_processes
	u_lower = -2.0				# bounds of the converted uniform distributions
	u_upper = 2.0
	z_cut = w_cut = 1.0

	# functor to convert chi squre to uniform square
	chisquare2uniformsquare = ChiSquare2UniformSquare()	

	p = Pool(n_processes)
	base_args = {'n_t':n_t, 'n_d':n_d, 'Beta':Beta, 'sigma2_e':sigma2_e,'C':C,
	             'b':b, 'chisquare2uniformsquare':chisquare2uniformsquare, 
			     'u_lower':u_lower, 'u_upper':u_upper, 'z_cut':z_cut, 'w_cut':w_cut}
	args_list = []
	for p_num in range(n_processes):
		seed = np.random.randint(100000000)
		process_args = deepcopy(base_args)
		process_args.update({'p_num':p_num, 'seed':seed})
		args_list.append(process_args)

	pis = p.map(do_n_tests_wrapper, args_list)
	pi = np.mean(pis)
	print 'pi:', pi

if __name__ == '__main__':
	main()