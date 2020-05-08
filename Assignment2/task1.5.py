import numpy as np
import matplotlib.pyplot as plt

cov_x = np.load('../covid_x.npy')
cov_t = np.load('../covid_t.npy')

########## ex. 1.5 #########

# array with days passed from the first (25 February)
x_days = np.arange(len(cov_x)).reshape(len(cov_x), 1)
# number of data points
N = len(cov_x)
# M = K + 2P
M = len(x_days)
gamma = int(10**6)
# matrix phi
phi = []
alpha = 0.005
beta = 0.009

w_map = []

# define Radial Basis Function
def rbf(x, c_m, sigma):
    result = np.exp((-(x-c_m)**2)/(2*(sigma**2)))
    return float(result)

for data_point in x_days:
    phi.append(gamma)
    # using M = K = 33
    for day in range(33):
        c_m = x_days[data_point] - x_days[day]
        phi.append(rbf(cov_t[data_point], c_m, 2.5))

phi = np.reshape(phi, (N, 34))

w_map = (np.linalg.pinv((phi.transpose() @ phi) + ((alpha/beta)*np.identity(34)))) @ phi.transpose() @ cov_t
y = phi @ w_map

plt.plot(cov_x, cov_t, 'x')
plt.plot(cov_x, y)
plt.show()

print(w_map)