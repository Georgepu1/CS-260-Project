import os
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.init as init
import numpy as np

from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error

# Get metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_logfile(filename, text):
    f = open(filename, 'w')
    f.write(text + "\n")
    f.close()


def log(filename, text):
    f = open(filename, 'a')
    f.write(text + "\n")
    f.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def get_subsample_dataset(trainset, subset):
    trainsubset = copy.deepcopy(trainset)
    trainsubset.data = [trainsubset.data[index] for index in subset]
    trainsubset.targets = [trainsubset.targets[index] for index in subset]
    return trainsubset


def get_subsample_dataset_label_noise(trainset, subset, noise_size):
    train_size = len(subset)
    trainsubset = copy.deepcopy(trainset)
    trainsubset.data = [trainsubset.data[index] for index in subset]
    trainsubset.targets = [trainsubset.targets[index] for index in subset]
    ######## shuffle
    shuffle_targets_subset = [trainsubset.targets[idx] for idx in range(train_size - noise_size, train_size)]
    shuffle(shuffle_targets_subset)
    for idx in range(train_size - noise_size, train_size):
        trainsubset.targets[idx] = shuffle_targets_subset[idx - train_size + noise_size]
    return trainsubset

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

dataset = 'mnist'
trial=5

# variables for bias-variance calculation
NUM_CLASSES = 10

# squared loss

# zero one loss

# train test curves over Number of trees (1, 10, 20) and
# number of leaves for each tree (10, 1000, 2000)

# or decision tree limited by the number of leaves?

# 10,000 training subset

ratio = [0.0, 0.2, 0.4, 0.6]
num_leaves = [10, 1000, 2000]
num_trees = [1, 10, 20]

train_subset = 10000
# set up random partition
permute_index = np.split(np.random.permutation(train_subset), trial)
NUM_CLASSES = 10
# Random forest and decision tree
for nt in num_trees:
    for nl in num_leaves:
        for r in ratio:
            TRAIN_ACC_SUM = 0.0
            TEST_ACC_SUM = 0.0

            MSE_TR_SUM = 0.0
            MSE_TE_SUM = 0.0
            ZO_TR_SUM = 0.0
            ZO_TE_SUM = 0.0

            outdir = '{}_trial{}_rf_noise{}'.format(dataset, trial, r)
            logfilename = os.path.join(outdir, 'log_rf_nt{}_nl{}-noise{}.txt'.format(nt, nl, r))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            init_logfile(logfilename, "trial\ttrain acc\ttest acc\ttrain mse\ttestmse\ttrainzo\ttestzo")
            print("Current ratio: {} and current complexity: nt {} nl {}".format(r, nt, nl))
            for t in range(trial):

                ##########################################
                # set up subsampled train loader
                ##########################################
                trainsubset = get_subsample_dataset_label_noise(trainset, permute_index[t], int(len(permute_index[t])*r))
                X_set = [np.array(x[0].reshape(28*28)) for x in trainsubset]
                y_set = [x[1] for x in trainsubset]
                X_testset = [np.array(x[0].reshape(28*28)) for x in testset]
                y_testset = [x[1] for x in testset]
                ##########################################
                # set up model and optimizer
                ##########################################
                net = RandomForestClassifier(n_estimators=nt, max_leaf_nodes=nl)
                net.fit(X_set, y_set)

                y_pred = net.predict(X_set)
                y_test_pred = net.predict(X_testset)

                train_acc = accuracy_score(y_set, y_pred)
                test_acc = accuracy_score(y_testset, y_test_pred)
                mse_tr = mean_squared_error(y_set, y_pred)
                mse_te = mean_squared_error(y_testset, y_test_pred)
                zo_tr = zero_one_loss(y_set, y_pred)
                zo_te = zero_one_loss(y_testset, y_test_pred)
                TRAIN_ACC_SUM += train_acc
                TEST_ACC_SUM += test_acc
                MSE_TR_SUM += mse_tr
                MSE_TE_SUM += mse_te
                ZO_TR_SUM += zo_tr
                ZO_TE_SUM += zo_te
                log(logfilename, "{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}".format(t, TRAIN_ACC_SUM / (t + 1), TEST_ACC_SUM / (t + 1),
                                                                      MSE_TR_SUM / (t + 1),
                                                                      MSE_TE_SUM / (t + 1),
                                                                      ZO_TR_SUM / (t + 1),
                                                                      ZO_TE_SUM / (t + 1)))
                print('trial: {}, train acc: {}, test acc: {}, train mse: {}, test mse: {}, train zo error: {}, test zo error: {}'.format(t, TRAIN_ACC_SUM / (t + 1), TEST_ACC_SUM / (t + 1),
                                                                      MSE_TR_SUM / (t + 1),
                                                                      MSE_TE_SUM / (t + 1),
                                                                      ZO_TR_SUM / (t + 1),
                                                                      ZO_TE_SUM / (t + 1)))

print('Program finished', flush=True)
