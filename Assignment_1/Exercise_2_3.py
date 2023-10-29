import numpy as np
import matplotlib.pyplot as plt

# Using a new seed to generate new data points
np.random.seed(99)

n = 50

mu_1 = np.array([4, 7])
cov_1 = np.array([[9, 3], [3, 10]])
c_1 = np.random.multivariate_normal(mu_1, cov_1, n)
c_1_noise = c_1 + np.random.normal(0, 1, c_1.shape)

mu_2 = np.array([5, 10])
cov_2 = np.array([[7, 0], [0, 16]])
c_2 = np.random.multivariate_normal(mu_2, cov_2, n)
c_2_noise = c_2 + np.random.normal(0, 1, c_2.shape)

# Classify using 10 sample classifier - function returns True if point belongs to Class 1 and False if point belongs to Class 2
def classifier_10(x_1, x_2):
    val = 0.4*x_1 + 6.28*x_2 - 52.16
    return val < 0

# Classify using 100 sample classifier - function returns True if point belongs to Class 1 and False if point belongs to Class 2
def classifier_100(x_1, x_2):
    val = 1.03*x_1 + 2.24*x_2 - 22.51
    return val < 0

# Classify new data using 10 sample classifier
correct_10 = 0
for i in range(len(c_1)):
    if classifier_10(c_1[i][0], c_1[i][1]):
        correct_10 += 1
    if not classifier_10(c_2[i][0], c_2[i][1]):
        correct_10 += 1

# Classify new data using 100 sample classifier
correct_100 = 0
for i in range(len(c_1)):
    if classifier_100(c_1[i][0], c_1[i][1]):
        correct_100 += 1
    if not classifier_100(c_2[i][0], c_2[i][1]):
        correct_100 += 1

print(f"The classification accuracy of the 5-sample classifier is {correct_10}% while the classification accuracy of the 100-sample classifier is {correct_100}%.")

# Plot the sample points for each class
plt.scatter(c_1[:,0], c_1[:,1], c='r')
plt.scatter(c_2[:,0], c_2[:,1], c='g')
x_1 = np.linspace(-5, 15, 100)
x_100 = -0.46*x_1+10.02
x_10 = -0.064*x_1+8.31

plt.plot(x_1, x_100, 'b-')
plt.plot(x_1, x_10, 'y-')
plt.legend(['Class 1', 'Class 2', '100-sample Classifier', '10-sample Classifier'])
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Decision Boundaries on Newly-Generated Noisy Samples')
plt.show()
