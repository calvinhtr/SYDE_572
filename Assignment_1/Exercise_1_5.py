import numpy as np
import matplotlib.pyplot as plt

# Calculate equiprobability contour equation
def equi_contour(x, mu, cov):
    mat = x - mu
    return (np.exp(-0.5 * np.sum(mat @ np.linalg.inv(cov) * mat, axis=2))*(1. / (np.sqrt(np.linalg.det(cov)*(np.pi*2)**len(mu)))))

# Using a seed to get the same results each time
np.random.seed(999)

# Exercise 1.5
n = 100

mu_1 = np.array([4, 7])
cov_1 = np.array([[9, 3], [3, 10]])
c_1 = np.random.multivariate_normal(mu_1, cov_1, n)

mu_2 = np.array([5, 10])
cov_2 = np.array([[7, 0], [0, 16]])
c_2 = np.random.multivariate_normal(mu_2, cov_2, n)

# Calculate sample mean 
samp_mu_1 = np.mean(c_1, axis = 0)
samp_mu_2 = np.mean(c_2, axis = 0)

# Calculate sample covariance matrix
samp_cov_1 = np.cov(c_1, rowvar=False)
samp_cov_2 = np.cov(c_2, rowvar=False)

# Calculate eigenvectors and eigenvalues of each covariance matrix
eig_vals_1, eig_vecs_1 = np.linalg.eigh(samp_cov_1)
eig_vals_2, eig_vecs_2 = np.linalg.eigh(samp_cov_2)

print("CLASS 1: \n The sample mean is:\n", samp_mu_1, "\n The sample covariance matrix is: \n", samp_cov_1, "\n The eigenvalues of the covariance matrix are: \n", eig_vals_1, "\n The corresponding eigenvectors are: \n", eig_vecs_1)
print("CLASS 2: \n The sample mean is:\n", samp_mu_2, "\n The sample covariance matrix is: \n", samp_cov_2, "\n The eigenvalues of the covariance matrix are: \n", eig_vals_2, "\n The corresponding eigenvectors are: \n", eig_vecs_2)

# Calculate contours and plot them for class 1
[X_1,X_2] = np.mgrid[-5:12.5:.01, -5:20:.01]
vals = np.dstack([X_1, X_2])
plt.figure()
contour_1 = equi_contour(vals, samp_mu_1, samp_cov_1)
plt.contour(X_1, X_2, contour_1, colors='r')
plt.scatter(c_1[:,0], c_1[:,1], c='r')
plt.title('Equiprobability Contours for Class 1')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')

# Calculate contours and plot them for class 2
[X_1,X_2] = np.mgrid[-5:15:.01, -5:25:.01]
vals = np.dstack([X_1, X_2])
plt.figure()
contour_2 = equi_contour(vals, samp_mu_2, samp_cov_2)
plt.contour(X_1, X_2, contour_2, colors='g')
plt.scatter(c_2[:,0], c_2[:,1], c='g')
plt.title('Equiprobability Contours for Class 2')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')

plt.show()
