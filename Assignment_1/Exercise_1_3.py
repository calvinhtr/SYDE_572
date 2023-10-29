import numpy as np
import matplotlib.pyplot as plt

# Exercise 1.3

# Calculate equiprobability contour equation
def equi_contour(x, mu, cov):
    mat = x - mu
    return (np.exp(-0.5 * np.sum(mat @ np.linalg.inv(cov) * mat, axis=2))*(1. / (np.sqrt(np.linalg.det(cov)*(np.pi*2)**len(mu)))))

# Define sample mean and covariance for each class
s_mean_1 = np.array([4.29, 4.88])
s_cov_1 = np.array([[7.13, 1.34], [1.34, 4.58]])

s_mean_2 = np.array([4.69, 11.16])
s_cov_2 = np.array([[13.74, 10.13], [10.13, 22.24]])

# Define samples for each class, from exercise 1.1

c_1 = np.array([
    [3.86317368, 6.98028738],
    [1.81099457, 6.01214467],
    [3.68588236, 3.72158188],
    [8.84837398, 5.98633342],
    [3.24418499, 1.70899736]
])

c_2 = np.array([
    [ 5.6974558,  18.28342712],
    [10.24735416, 12.71358796],
    [ 3.7893197,  10.94008982],
    [ 3.60590629,  7.08063391],
    [ 0.09702069,  6.7808758 ]
])

# Create a grid of points
[X_1,X_2] = np.mgrid[-5:15:.01, -5:25:.01]
vals = np.dstack([X_1, X_2])

# Calculate contours and plot them
contour_1 = equi_contour(vals, s_mean_1, s_cov_1)
plt.contour(X_1, X_2, contour_1, colors='r')

contour_2 = equi_contour(vals, s_mean_2, s_cov_2)
plt.contour(X_1, X_2, contour_2, colors='g')

# Plot the sample points for each class
plt.scatter(c_1[:,0], c_1[:,1], c='r')
plt.scatter(c_2[:,0], c_2[:,1], c='g')

# Label graph and axes
plt.title('Equiprobability Contours for Class 1 and 2')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend(['Class 1', 'Class 2'])

plt.show()




