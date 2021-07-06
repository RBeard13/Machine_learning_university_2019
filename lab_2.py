import numpy as np

n = 1000
x = np.linspace(0, np.pi, n)
y = 100 * np.sin(x) + 0.5 * np.exp(x) + 300
err = np.random.rand(n) * 10
labels = y + err

errors = []
q_d = []
plana = np.ones((n, 1))
for i in range(1, 11):
    plana_new_axis = (x ** i).reshape(-1, 1)
    plana = np.hstack((plana, plana_new_axis))

    w = np.dot(np.dot(np.linalg.inv(np.dot(plana.T, plana)), plana.T), labels)

    reg = 0
    for l in range(len(w)):
        reg += w[l] * x ** l

    err = np.sum((labels - reg)**2)
    errors.append(err)
    q_d.append(i)


import numpy as np
from itertools import combinations
from sklearn.utils import shuffle

n = 1000
n1 = 800
n2 = 100
x = np.linspace(0, np.pi, n)
y = 100 * np.sin(x) + 0.5 * np.exp(x) + 300
err = np.random.rand(n) * 10
labels = y + err
x, labels = shuffle(x, labels)
tr_data = x[:n1]
tr_labels = labels[:n1]
val_data = x[n1:n1+n2]
val_labels = labels[n1:n1+n2]
test_data = x[n1+n2:]
test_labels = labels[n1+n2:]
func = [np.cos, np.sin, np.exp, np.sqrt]
errors = []
w_by_errors = []
f_by_errors = []
for q in range(1, 4):
    c = combinations(func, q)
    for f in c:
        plana = np.ones((n1, 1))
        for l in range(len(f)):
            plana_new_axis = f[l](tr_data).reshape(-1, 1)
            plana = np.hstack((plana, plana_new_axis))
        w = np.dot(np.dot(np.linalg.inv(np.dot(plana.T, plana)), plana.T), tr_labels)

        reg = w[0]
        for k in range(len(w) - 1):
            reg += w[k+1] * f[k](val_data)

        err = np.sum((val_labels - reg)**2)
        errors.append(err)
        w_by_errors.append(w)
        f_by_errors.append(f)

errors = np.array(errors)
min_errors = np.argsort(errors)[:4]
test_err = []
test_w = []
test_f = []
for i in min_errors:
    w = w_by_errors[i]
    f = f_by_errors[i]

    reg = w[0]
    for k in range(len(w) - 1):
        reg += w[k+1] * f[k](test_data)

    err = np.sum((test_labels - reg)**2)
    test_err.append(err)
    test_w.append(w)
    test_f.append(f)

test_err = np.array(test_err)
ind = np.argmin(test_err)
best_err = test_err[ind]
best_w = test_w[ind]
best_f = test_f[ind]

print(best_err, best_f, best_w)




















