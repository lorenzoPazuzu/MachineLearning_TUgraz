import numpy as np
import matplotlib.pyplot as plt

cov_x = np.load('../covid_x.npy')
cov_t = np.load('../covid_t.npy')
fig = plt.figure(figsize=(15,8))

# prediction
P = 20
# (past) days to train the model
K = 50
# array with days passed from the first (also future days)
x_days = np.arange(K+P).reshape(K+P, 1)
# M = K + 2P
M = K + 2*P
gamma = int(10**6)
# matrix phi
phi = []
phi_pred = []
y_dist = []
sig = []
w_map = []

# hyperparameters
alpha = 0.0005
beta = 0.0009
sigma_r_2 = 50


# define Radial Basis Function
def rbf(x, c_m, sigma_sq):
    result = np.exp(-((x-c_m)**2)/(2*(sigma_sq)))
    return result

# normal distribution
def normal(x, mu, sigma):
    result = (1/(np.sqrt(2*np.pi)*sigma))*np.exp((-1/2)*((x-mu)/sigma)**2)
    return result

# design matrix
for data_point in range(K):
    phi.append(gamma)
    # M = K + 2P
    for day in range(M):
        c_m = day
        phi.append(float(rbf(x_days[data_point], c_m, sigma_r_2)))


phi = np.reshape(phi, (K, M+1))

# optimal MAP
w_map = np.linalg.pinv((phi.transpose() @ phi) + ((alpha/beta)*np.identity(M+1))) @ phi.transpose() @ cov_t[:K]


# prediction phi_pred
for data_point in range(K+P):
    phi_pred.append(gamma)
    # M = K + 2P
    for day in range(M):
        c_m = day
        phi_pred.append(float(rbf(x_days[data_point], c_m, sigma_r_2)))

phi_pred = np.reshape(phi_pred, (K+P, M+1))

#appending new dates:
cov_x = np.append(cov_x, ['16.04', '17.04', '18.04', '19.04', '20.04', '21.04', '22.04', '23.04', '24.04','25.04', '26.04', '27.04','28.04', '29.04', '30.04','01.05', '02.05', '03.05','04.05', '05.05', '06.05','07.05', '08.05', '09.05'])

# regression curve
y_pred = phi_pred @ w_map

# complete predictive distribution
S = np.linalg.pinv(alpha*np.identity(M+1) + beta * (phi.transpose() @ phi))
m = beta * (S @ phi.transpose() @ cov_t[:K])

lin_days = np.arange(0, len(x_days), step=1)

#drawing area +/-2sigma 
for index in range(len(phi_pred[:, 0])):
    y_dist = np.append(y_dist, m.transpose()@phi_pred[index])
    sig = np.append(sig, np.sqrt(1/beta + phi_pred[index].transpose() @ S @ phi_pred[index]))

plt.fill_between(lin_days, y_dist+2*sig, y_dist-2*sig, color='limegreen', alpha = 0.2, label = '+/- 2sigma')

#drawing M kernels
x_cont = np.arange(0, K+P, step = 0.1)
for day in range(M):
        plt.plot(x_cont, rbf(x_cont, day, sigma_r_2) * w_map[day + 1], alpha=0.3, color ='b')



plt.plot(x_days[:K], cov_t[:K], '+', color = 'black', label = 'Infections per day')
plt.xlabel("Date")
plt.ylabel("Infected")
plt.plot(cov_x[:(K+P)], y_pred, label = 'RBF-Bayesian Linear reg.')
plt.plot(cov_x[:(K+P)], y_pred, '.', color = 'black')
plt.xticks(cov_x[np.arange(0, K+P, step = 1)], rotation=45)
plt.legend()
x_assis = np.arange(0,1, step=0.01)
plt.title('sigma_r_2 = %f alpha = %f beta = %f K = %d' % (sigma_r_2, alpha, beta, K))

#plt.savefig('plots/17.pdf')
plt.show()