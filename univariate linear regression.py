import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')


# the main challenge is to find right values for w and b that fit the training set data
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


# size of the house
x_train = np.array([1.0, 2.0])
# price of the house
y_train = np.array([300.0, 500.0])
print(f"x_train.shape: {x_train.shape}")
print(np.zeros(2))
tmp_f_wb = compute_model_output(x_train, 200, 100, )
print(tmp_f_wb)
plt.scatter(x_train, y_train, marker='x', color='r')
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.show()
