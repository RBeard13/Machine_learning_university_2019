
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 1000)
gt = 100 * np.sin(x) + 0.5 * np.exp(x) + 300

eps = np.random.randn(1000)*10

data = gt + eps

plt.plot(x, gt,'b--', alpha = 0.8, label = 'func')
plt.scatter(x, data, 10, 'g', alpha=0.8, label='data')
#plt.plot(x, data)
#plt.plot(x, upper)
plt.show()
#plt.plot(x, lower)
plana = np.ones((1000, 1))
error_x = []
error_y = []

for er in range(1, 21):
    new_column = (x ** er).reshape(1000, 1)
    plana = np.concatenate((plana, new_column), axis = 1)

    w = np.dot(np.dot(np.linalg.inv(np.dot(plana.T, plana)), plana.T), data)

    reg = 0
    for i in range(len(w)):
        reg = reg + w[i] * x ** i
    error_x.append(sum((data - reg)**2))
    error_y.append(er)
    print(er, '   ', sum((data - reg)**2))
    ax = plt.subplot(4, 5, er)
    ax.plot(x, data, 'ro', x, reg, 'b')
    #plt.plot(x, reg, label = 'reg')
#plt.legend(loc='upper right', prop={'size': 20})
#plt.title('Test')
#plt.show()
plt.plot(error_y, error_x)
plt.show()
