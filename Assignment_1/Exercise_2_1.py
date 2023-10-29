import numpy as np
import matplotlib.pyplot as plt

# Initialize datasets
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

# Plot the sample points for each class
plt.scatter(c_1[:,0], c_1[:,1], c='r')
plt.scatter(c_2[:,0], c_2[:,1], c='g')

# Plot decision boundary for each class
x_1 = np.linspace(0, 15, 100)
x_2 = -0.064*x_1+8.31

plt.plot(x_1, x_2, 'b-')
plt.title('MED Decision Boundary')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend(['Class 1', 'Class 2', 'Decision Boundary'])

plt.show()