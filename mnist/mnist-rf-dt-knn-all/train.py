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
# Track experiments
current_train_metrics = []


ratio = [0.0, 0.2, 0.4, 0.6]
model_width = [5, 10, 15, 20, 25, 30, 35, 40, 60, 80, 100]
# set up index after random permutation
permute_index = np.split(np.random.permutation(len(trainset)), trial)
NUM_CLASSES = 10
# Random forest and decision tree
for class_type in range(3):
    if class_type == 0:
        print('--NEW TYPE: RANDOM FOREST--')
        current_train_metrics.append('--NEW TYPE: RANDOM FOREST--')
    elif class_type == 1:
        print('--NEW TYPE: Decision Tree--')
        current_train_metrics.append('--NEW TYPE: Decision Tree--')
    else:
        print('--NEW TYPE: KNN--')
        current_train_metrics.append('--NEW TYPE: KNN--')

    for o_ind in range(len(model_width)):
        for ind in range(len(ratio)):
            TRAIN_ACC_SUM = 0.0
            TEST_ACC_SUM = 0.0
            TRAIN_LOSS_SUM = 0.0
            TEST_LOSS_SUM = 0.0
            outdir = None
            logfilename = None
          if class_type == 0:
              outdir = '{}_trial{}_rf_noise{}'.format(dataset, trial, ratio[ind])
              logfilename = os.path.join(outdir, 'log_rf_width{}-noise{}.txt'.format(model_width[o_ind], ratio[ind]))
          elif class_type == 1:
              outdir = '{}_trial{}_dt_noise{}'.format(dataset, trial, ratio[ind])
              logfilename = os.path.join(outdir, 'log_dt_width{}-noise{}.txt'.format(model_width[o_ind], ratio[ind]))
          else:
              outdir = '{}_trial{}_knn_noise{}'.format(dataset, trial, ratio[ind])
              logfilename = os.path.join(outdir, 'log_knn_width{}-noise{}.txt'.format(model_width[o_ind], ratio[ind]))

          if not os.path.exists(outdir):
              os.makedirs(outdir)

          print("Current ratio: {} and current complexity: {}".format(ratio[ind], model_width[o_ind]))
          init_logfile(logfilename, "trial\ttrain acc\ttest acc")
          current_train_metrics.append('--Break--')
          for t in range(trial):
              ##########################################
              # set up subsampled train loader
              ##########################################
              trainsubset = get_subsample_dataset_label_noise(trainset, permute_index[t], int(len(permute_index[t])*ratio[ind]))
              X_set = [np.array(x[0].reshape(28*28)) for x in trainsubset]
              y_set = [x[1] for x in trainsubset]
              X_testset = [np.array(x[0].reshape(28*28)) for x in testset]
              y_testset = [x[1] for x in testset]
              ##########################################
              # set up model and optimizer
              ##########################################
              if class_type == 0:
                  net = RandomForestClassifier(n_estimators=model_width[o_ind])
              elif class_type == 1:
                  net = DecisionTreeClassifier(criterion="entropy", max_depth=model_width[o_ind])
              else:
                  net = KNeighborsClassifier(n_neighbors=10)
              net.fit(X_set, y_set)
              y_pred = net.predict(X_set)
              train_acc = accuracy_score(y_set, y_pred)

              y_test_pred = net.predict(X_testset)
              test_acc = accuracy_score(y_testset, y_test_pred)

              TRAIN_ACC_SUM += train_acc
              TEST_ACC_SUM += test_acc
              log(logfilename, "{}\t{:.5}\t{:.5}".format(t, TRAIN_ACC_SUM / (t + 1), TEST_ACC_SUM / (t + 1)))
              print('trial: {}, train acc: {}, test acc: {}'.format(t, TRAIN_ACC_SUM / (t + 1), TEST_ACC_SUM / (t + 1)))
              current_train_metrics.append('trial: {}, train acc: {}, test acc: {}'.format(t, train_acc, test_acc))
      if class_type == 2:
          print("KNN only has one parameter, no complexity")
          break

print('Program finished', flush=True)

# import pickle as pkl
# pkl.dump(current_train_metrics, open('output-train.pkl', 'wb'))
# import numpy as np
# from numpy import savetxt

# np_train_metrics = np.array(current_train_metrics)

