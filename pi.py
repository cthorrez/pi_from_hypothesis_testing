import numpy as np 
import matplotlib.pyplot as plt




def get_dataset(n, Beta0, Beta1, Beta2, sigma2_e):
	# generate 2 columns of random numbers
	x = np.random.rand(n,2)
	ones = np.ones([n,1])
	x = np.hstack([ones,x])

	y = np.dot(x,np.array([[Beta0,Beta1,Beta2]]).T) 

	e = np.random.normal(loc=0, scale=np.sqrt(sigma2_e), size=[n,1])

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


	






def main():
	# initialize linear equation: t = 2*x1 + 5*x2 + 0 (no bias term)
	Beta0, Beta1, Beta2 = np.array([0,1,-1], dtype=float)
	sigma2_e = 1.0

	z_reject = 0
	w_reject = 0
	z_total = 0
	w_total = 0

	w_not_z = 0

	# C and b specify constraints that Beta1_hat = Beta1 and Beta2_hat = Beta2
	C = np.array([[0,1,0],
		 		 [0,0,1]])
	b = np.array([[Beta1,Beta2]]).T
		

	ws = []
	z1s = []
	z2s = []

	iters = 1000
	for i in range(iters):
		if i % (iters/100) == 0:
			print i/float(iters)*100, '%'

		w_rejected = False
		z_rejected = False

		fake_Beta1 = np.random.rand()
		fake_Beta2 = np.random.rand()
		x,t =  get_dataset(1000, Beta0, Beta1, Beta2, sigma2_e)


		Beta_hat = get_ols_estimator(x,t)

		z_Beta1 = get_z_statistic(x,t,Beta_hat,1,Beta1)
		#z1s.append(z_Beta1[0])
		z_Beta2 = get_z_statistic(x,t,Beta_hat,2,Beta2)
		#z2s.append(z_Beta2[0])
		w = get_wald_statistic(x,t,Beta_hat,C,b)
		#ws.append(w[0,0])
		#w2 = get_wald_statistic2(x,t,Beta_hat,C,b)

		#zcut = 1.217
		zcut = 1.565
		zcut = 1.0

		if (abs(z_Beta1) > zcut) or (abs(z_Beta2) > zcut):
			z_rejected = True
			z_reject += 1

		z_total += 1

		# wcut = 5.991
		wcut = 1.0
		if w > wcut:
			w_reject += 1
			w_rejected = True

		w_total += 1

		if w_rejected and not z_rejected:
			w_not_z += 1


	z_reject_rate = z_reject/float(z_total)
	w_reject_rate = w_reject/float(w_total)

	print('z reject rate: {}'.format(z_reject_rate))
	print('w reject rate: {}'.format(w_reject_rate))


	print w_not_z

	pi = -1*(4*((w_not_z * 2.95615989851) / iters) - 4)
	print('estimate of pi: {}'.format(pi))

	#n, bins, patches = plt.hist(z1s, 50, density=True, facecolor='g', alpha=0.75)


	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('idk lol')
	plt.grid(True)
	plt.show()








if __name__ == '__main__':
	main()