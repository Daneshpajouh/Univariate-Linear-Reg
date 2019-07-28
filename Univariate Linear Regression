%reset
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
x = np.matrix((np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Univariate/master/x_train.csv')), skiprows = 1)))
y = np.matrix(np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Univariate/master/y_train.csv')), skiprows = 1))
x_test = np.matrix((np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Univariate/master/x_test.csv')), skiprows = 1)))
y_test = np.matrix(np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Univariate/master/y_test.csv')), skiprows = 1))
alpha = 0.0001
theta_0 = 0
theta_1 = 0
num_itr = 50
cost = []
m = len(y.transpose())
epsilon = 1e-1

for i in range(num_itr):              # Gradient Descent function ( Training )
    h_x = theta_0 + theta_1 * x
    error = float(np.sum(np.power(np.array((h_x - y)),2), axis = 1)) / (2*m)
    cost.append(error)
    theta_0 = theta_0 - (((h_x - y).sum(axis = 1))*(alpha/m))
    theta_1 = theta_1 - (((np.multiply((h_x - y),x)).sum(axis = 1))*(alpha/m))
    print(error)
    if i < 5:
        pass
    elif abs(np.mean(cost[-5:]) - error) <= epsilon:
        break

hypo = theta_0 + theta_1 * x_test     # Testing the parameters with test values        
print('\n\nFinal Theta Value is:',theta_0, theta_1, '\n\nAlpha value is:', alpha, '\n\nFinal Error is:', error, "\n\n", hypo)

fig, ax = plt.subplots()                    # Cost function figure
ax.scatter(range(len(np.array(cost).transpose())), np.array(cost).transpose(), marker="x", c="red")
plt.title("Cost", fontsize=16)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.axis([0, int(len(np.array(cost).transpose())), 0 , float(cost[0])])
ax.plot(range(len(np.array(cost).transpose())), np.array(cost).transpose(), linewidth=2)
fig

fig, ax = plt.subplots()                    # Test Dataset figure
ax.scatter([x_test], [y_test.transpose()], marker="x", c="red")
plt.title("Dataset", fontsize=16)
plt.xlabel("Input Dataset", fontsize=14)
plt.ylabel("Output Dataset", fontsize=14)
plt.axis([-10, 125, -10, 125])
ax.plot(x_test.transpose(), hypo.transpose(), linewidth=2)
fig
