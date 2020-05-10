import numpy as np
import matplotlib.pyplot as plt

cov_x = np.load('../covid_x.npy')
cov_t = np.load('../covid_t.npy')

# prediction
P = 20
# (past) days to train the model
K = 33
# array with days passed from the first (also future days)
x_days = np.arange(K+P).reshape(K+P, 1)
# M = K + 2P
M = K + 2*P
gamma = int(10**6)
# matrix phi
phi = []
phi_pred = []
sigma_r = 2.5
y_dist = []
sig = []
alpha = 1
beta = 10

w_map = []

# define Radial Basis Function
def rbf(x, c_m, sigma):
    result = np.exp(-((x-c_m)**2)/(2*(sigma**2)))
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
        phi.append(float(rbf(x_days[data_point], c_m, sigma_r)))


phi = np.reshape(phi, (K, M+1))

# optimal MAP parameters
w_map = np.linalg.pinv((phi.transpose() @ phi) + ((alpha/beta)*np.identity(M+1))) @ phi.transpose() @ cov_t[:K]


# prediction phi_pred
for data_point in range(K+P):
    phi_pred.append(gamma)
    # M = K + 2P
    for day in range(M):
        c_m = day
        phi_pred.append(float(rbf(x_days[data_point], c_m, sigma_r)))

phi_pred = np.reshape(phi_pred, (K+P, M+1))

cov_x = np.append(cov_x, ['16.04', '17.04', '18.04'])

# MAP curve
y_pred = phi_pred @ w_map

# complete predictive distribution
S = np.linalg.pinv(alpha*np.identity(M+1) + beta * (phi.transpose() @ phi))
m = beta * (S @ phi.transpose() @ cov_t[:K])

lin_days = np.arange(0, len(x_days), step=1)


for index in range(len(phi_pred[:, 0])):
    y_dist = np.append(y_dist, m.transpose()@phi_pred[index])
    sig = np.append(sig, np.sqrt(1/beta + phi_pred[index].transpose() @ S @ phi_pred[index]))

plt.fill_between(lin_days, y_dist+2*sig, y_dist-2*sig, color='turquoise', alpha = 0.5)

#drawing M kernels
x_cont = np.arange(0, 53, step = 0.1)
for day in range(M):
        plt.plot(x_cont,rbf(x_cont, day, sigma_r)*w_map[day+1], alpha=0.3, color = 'b')



plt.plot(x_days[:K], cov_t[:K], '+', color = 'black')
plt.xlabel("Date")
plt.ylabel("Infected")
plt.plot(cov_x, y_pred)
plt.xticks(cov_x[np.arange(0, 53, step = 3)], rotation=45)

x_assis = np.arange(0,1, step=0.01)
plt.title('sigma_r = %f alpha = %f beta = %f' % (sigma_r, alpha, beta))
#plt.savefig('plots/7.pdf')
plt.show()
