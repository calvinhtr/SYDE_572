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

# Create noisy data
np.random.seed(99)

n = 50

c_1_new = np.random.multivariate_normal(mu_1, cov_1, n)
c_1_noise = c_1_new + np.random.normal(0, 1, c_1_new.shape)

c_2_new = np.random.multivariate_normal(mu_2, cov_2, n)
c_2_noise = c_2_new + np.random.normal(0, 1, c_2_new.shape)

accuracy = []

for k in range(1,6):
    correct = 0
    for i in range(len(c_1_noise)):
        if knn(c_1, c_2, c_1_noise[i], k):
            correct += 1
        if not knn(c_1, c_2, c_2_noise[i], k):
            correct += 1
    accuracy.append(correct)
# Plot accuracies for each k-value
plt.figure()
plt.plot([k for k in range(1,6)], accuracy)

# Labels
plt.title(f"Accuracy for each k-value")
plt.xlabel("k")
plt.ylabel("Accuracy (%)")

plt.show()
