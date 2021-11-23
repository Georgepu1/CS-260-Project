# Souce: https://colab.research.google.com/drive/1MTk5fHJf3eG5LAE2VlvHM-h49f36DAVd?usp=sharing#scrollTo=RncETEOQDAoU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
np.random.seed(0)

def train(coeff, Xt, Xv, yt, yv):
    p = lambda x: PolynomialFeatures(coeff, interaction_only=True).fit_transform(x)
    return 1 - Perceptron().fit(p(Xt), yt).score(p(Xv), yv)

#Xt, Xv, yt, yv = train_test_split(*make_moons(10000, noise=0.4), test_size=0.3)

complex = list(range(1, 10, 1))
noise = np.geomspace(0.1, 1.0, num=6)
#cx, cy = np.meshgrid(noise, complex)
z = np.zeros((len(noise), len(complex)))
for i, n in enumerate(noise):
    for j, c in enumerate(complex):
        print(str(c), end=' ')
        # corrupt same datapoint - or make sklearn draw form same seed?
        # cache dataset, corrupt
        # plot accuracy
        z[i][j] = train(c, *train_test_split(*make_classification(100000, 10, n_informative=5, class_sep=0.1, n_redundant=0, flip_y=n), test_size=0.20))

PolynomialFeatures(3, include_bias=False).fit_transform([[2, 0]])

fig = plt.figure(figsize = (8,8))
ax = plt.axes()
for i, n in enumerate(noise):
    ax.plot(complex, z[i,:], label=f"{n:.5f}% noisy")
ax.plot([5.0] * 7, np.arange(0.25, 0.6, 0.05), '--')
plt.xlabel('complexity')
plt.ylabel('loss')
plt.legend()
plt.show()
# surf = ax.plot_surface(cx, cy, z, cmap=mpl.cm.coolwarm,
                       #linewidth=0, antialiased=False)

#for angle in range(0, 360):
#   ax.view_init(angle,30)
#   plt.draw()
#   plt.pause(.001)

Xt, Xv, yt, yv = train_test_split(*make_moons(noise=0.1), test_size=0.1)
test = Perceptron(random_state=0, fit_intercept=False)
test.fit(PolynomialFeatures(1, include_bias=False).fit_transform(Xt), yt)

# NOTE: want one moon dataset with the same amount of noise, then gradually corrupt the labels 1) symetrically (both pos to neg and neg to pos) and 2) asymetrically (only pos to neg) and look at how decision bounds change

# test.intercept_

xr = np.concatenate((np.linspace(-1, 1).reshape(50, 1), np.ones((50, 1))), axis=1)
plt.plot(xr[:,0], np.dot(xr, test.coef_.T))

# def get_decision_bound(weights, coeff, start, end):
#     n = 100
#     xr = np.linspace(start, end, num=n).reshape(n, 1)
#     xblown = np.concatenate((#something sklearn ployfeatures, np.ones((n, 1)), axis=1)
#     return xr, np.dot(xblown, weights.T)
