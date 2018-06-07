In this project I estimate the valu of pi using statistical tests.
I generate data sets with set linear coefficients and then train OLS regression on them.
I then perform tests on the regression estimated coefficients.
I compute the wald test statistic with the dual hypothesis that each coefficient is the true value.
Then I compute z-statistics for each of the two coefficients independently.
For the wald test, I reject if the test statistic is greater than 1. 
For the z tests I reject if either one is greater than 1.
Thus the acceptance region for the z tests is a square in 2 dimension z space.
The wald test statistic lives in chi-squared space with two degrees of freedom.
This means that it is the sum of two squared gaussians. With that in mind, the
equation for a circle arises. x^2 + y^2 = 1 so the acceptance region is a circle
exactly inscribed in the z test's square region. 
I generate many such synthetic data sets and keep track of the fraction of which
are rejected by the wald test but accepted by the z tests. These correspond to
points inside the square, but outside of the circle. I multiply this fraction
by an odds ratio which represents the epected number of points that would fall
in that region sampled from a bivariate Gaussian. After this ajustment I can use the ratio
of the areas of a circle and square to extract pi.

With 100,000,000 synthetic datasets of 1000 points each with Gaussian noise of 1.0,
my program returned an estimated value for pi of 3.14141138187. 

Full writeup at cthorrez.github.io