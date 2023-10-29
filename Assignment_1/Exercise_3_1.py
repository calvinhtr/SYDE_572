import numpy as np
import matplotlib.pyplot as plt

def euc_dist(point_1, point_2):
    return np.sum((point_1-point_2)**2)**0.5

# Returns True if point is in class 1, returns False if point is in class 2
def knn(class_1, class_2, point, k = 1) -> bool:
    # Determine distances to k-nearest neighbors for each class
    d_1 = [euc_dist(p_1, point) for p_1 in class_1]
    d_2 = [euc_dist(p_2, point) for p_2 in class_2]

    ind_1 = np.argsort(d_1)
    ind_2 = np.argsort(d_2)

    counter_1, counter_2 = 0, 0
    for i in range(k):
        if d_1[ind_1[counter_1]] < d_2[ind_2[counter_2]]:
            counter_1 += 1
        else:
            counter_2 += 1

    return counter_1 >= counter_2

# Using a seed to get the same results as from Exercise 1
np.random.seed(999)

n = 100

mu_1 = np.array([4, 7])
cov_1 = np.array([[9, 3], [3, 10]])
c_1 = np.random.multivariate_normal(mu_1, cov_1, n)

mu_2 = np.array([5, 10])
cov_2 = np.array([[7, 0], [0, 16]])
c_2 = np.random.multivariate_normal(mu_2, cov_2, n)

# Create meshgrid to make plots
x_min = min(c_1[:,0].min()-1, c_2[:,0].min()-1)
x_max = max(c_1[:,0].max()+1, c_2[:,0].max()+1)
y_min = min(c_1[:,1].min()-1, c_2[:,1].min()-1)
y_max = max(c_1[:,1].max()+1, c_2[:,1].max()+1)
xv, yv = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

for k in range(1,6):
    res = np.array([])

    # Determine class for each point
    for x, y in zip(xv.ravel(), yv.ravel()):
        res = np.append(res, knn(c_1, c_2, [x, y], k))
    
    res = res.reshape(xv.shape)

    # Plot for each k-value
    plt.figure()
    plt.scatter(c_1[:,0], c_1[:,1], c='b')
    plt.scatter(c_2[:,0], c_2[:,1], c='r')
    plt.contourf(xv, yv, res, alpha=0.2, cmap=plt.cm.RdBu)
    
    # Labels
    plt.title(f"{k}-Nearest Neighbor Classifier")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.legend(["Class 1", "Class 2"])

    plt.show()
