# Source: https://colab.research.google.com/drive/18QprykltuEVvguAtRfreuCQeXdTa5gv_?usp=sharing
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import pi as PI
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def training_with_label_noise(noise_level, k_start, NUM_MODELS, y_train_raw, eps):

    # add noise to training data
    y_train = y_train_raw + noise_level * eps

    train_mse = []
    test_mse = []

    for k in range(k_start, k_start+NUM_MODELS):

      # k-th degree polynomial coefficients
      fit_coeff = np.polyfit(x_train, y_train, deg = k)

      # train and test k-th degree polynomial fit
      y_train_pred = np.polyval(fit_coeff, x_train)
      y_test_pred = np.polyval(fit_coeff, x_test)

      # train and test MSE of k-th degree polynomial fit
      iter_train_mse = mean_squared_error(y_train_pred, y_train)
      iter_test_mse = mean_squared_error(y_test_pred, y_test)

      train_mse.append(iter_train_mse)
      test_mse.append(iter_test_mse)

    best_degree = k_start+test_mse.index(min(test_mse))
    print("Best fit polynomial degree: ", best_degree)

    return train_mse, test_mse, best_degree, y_test_pred, y_train_pred, y_train

np.random.seed(0)

plt.style.use('ggplot')

# number of observations
NUM_OBS = 200

# predictors
x = np.linspace(0, 2, num = NUM_OBS)
# noise
eps = np.random.normal(0, 1, NUM_OBS)
# outcome
y = np.sin(PI*x) + 0.000001*eps

x_train, x_test, y_train_raw, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.4,
                                                    random_state = 1)

# number of polynomial models to investigate
NUM_MODELS = 11
k_start = 3
eps = np.random.normal(0, 1, y_train_raw.shape[0])
list_noise = [0.0001*i for i in range(5)] + [0.001*i for i in [1, 5]] + [0.01*i for i in [1, 2, 3, 4, 6, 10, 12]]
# list_noise = [np.log(i) for i in np.linspace(1, np.e**0.1, 50)]


x_degree = range(k_start,k_start+NUM_MODELS)
best_degree = []

ftsize = 20
plt.figure()
fig, axs = plt.subplots(1, 3, figsize = (23, 8))
# axs[0].set_title('Train Set MSE')
# axs[1].set_title('Test Set MSE')
# axs[2].set_title('Best Polynomial Degree')

axs[0].set_yscale('log')
axs[1].set_yscale('log')

axs[0].set_xlabel("Polynomial Degree", fontsize=ftsize)
axs[1].set_xlabel("Polynomial Degree", fontsize=ftsize)
axs[2].set_xlabel("Noise Level", fontsize=ftsize)

axs[0].set_ylabel("Train Set MSE", fontsize=ftsize)
axs[1].set_ylabel("Test Set MSE", fontsize=ftsize)
axs[2].set_ylabel("Best Polynomial Degree", fontsize=ftsize)

y_test_pred = []
y_train = []
y_train_pred =[]

for noise_level in list_noise:

    result = training_with_label_noise(noise_level, k_start, NUM_MODELS, y_train_raw, eps)
    label = '{}'.format(noise_level)
    if len(label) > 6:
      label = label[0: 6]
    axs[0].plot(x_degree, result[0], label = label, marker='o')
    axs[1].plot(x_degree, result[1], label = label, marker='o')
    best_degree.append(result[2])

    if noise_level == 0 or noise_level == 0.02 or noise_level == 0.1:
        y_test_pred.append([noise_level, result[3]])
        y_train_pred.append([noise_level, result[4]])
        y_train.append([noise_level, result[5]])

# axs[2].set_xscale('log')
axs[2].plot(list_noise, best_degree, marker='o')
# axs[2].set_xlim(0, None)

axs[0].legend(fontsize=10)
axs[1].legend(fontsize=10)

plt.figure()
fig, axs = plt.subplots(2, len(y_test_pred), figsize = (20, 10))

cnt = 0
for idx in range(len(y_test_pred)):

    axs[0, cnt].scatter(x_train, y_train[idx][1], label = 'label')
    axs[0, cnt].scatter(x_train, y_train_pred[idx][1], label = 'prediction')
    axs[0, cnt].legend()

    axs[1, cnt].scatter(x_test, y_test, label = 'label')
    axs[1, cnt].scatter(x_test, y_test_pred[idx][1], label = 'prediction')
    axs[1, cnt].legend()
    cnt = cnt + 1


