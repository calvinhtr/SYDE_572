import numpy as np
import matplotlib.pyplot as plt

# Using a seed to get the same results as from Exercise 1
np.random.seed(999)

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

# Calculate decision boundary
diff = samp_mu_1-samp_mu_2
m = -diff[0]/diff[1]
b = 0.5*(np.matmul(np.transpose(samp_mu_1), samp_mu_1)-np.matmul(np.transpose(samp_mu_2), samp_mu_2))/diff[1]
print(f"The equation representing the decision boundary is x_2 = {m:.2f}x_1 + {b:.2f}")

# Plot the sample points for each class
plt.scatter(c_1[:,0], c_1[:,1], c='r')
plt.scatter(c_2[:,0], c_2[:,1], c='g')

# Plot decision boundary for each class
x_1 = np.linspace(-5, 15, 100)
x_2 = -0.46*x_1+10.04

plt.plot(x_1, x_2, 'b-')
plt.title('MED Decision Boundary')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend(['Class 1', 'Class 2', 'Decision Boundary'])

plt.show()