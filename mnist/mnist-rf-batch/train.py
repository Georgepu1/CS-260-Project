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
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

dataset = 'mnist'
trial=5
lr=0.1
lr_decay=100
num_epoch=10
test_size=10000
save_freq=100

# variables for bias-variance calculation
NUM_CLASSES = 10
OUTPUST_SUM = torch.Tensor(test_size, NUM_CLASSES).zero_()
OUTPUTS_SUMNORMSQUARED = torch.Tensor(test_size).zero_()

# Track experiments
current_train_metrics = []


ratio = [0.0, 0.2, 0.4, 0.6]
model_width = [25, 50, 100, 200, 400, 800, 1000, 1600, 2000]
# set up index after random permutation
permute_index = np.split(np.random.permutation(len(trainset)), trial)
NUM_CLASSES = 10

for o_ind in range(len(model_width)):
    for ind in range(len(ratio)):
        TRAIN_ACC_SUM = 0.0
        TEST_ACC_SUM = 0.0
        TRAIN_LOSS_SUM = 0.0
        TEST_LOSS_SUM = 0.0
        outdir = ''
        outdir = '{}_trial{}_rf_noise{}'.format(dataset, trial, ratio[ind])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print("Current ratio: {} and current complexity: {}".format(ratio[ind], model_width[o_ind]))
        logfilename = os.path.join(outdir, 'log_width{}-noise{}.txt'.format(model_width[o_ind], ratio[ind]))
        init_logfile(logfilename, "trial\ttrain acc\ttest acc")
        current_train_metrics.append('--Break--')
        for t in range(trial):
            ##########################################
            # set up subsampled train loader
            ##########################################
            trainsubset = get_subsample_dataset_label_noise(trainset, permute_index[t], int(len(permute_index[t])*ratio[ind]))
            trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=128, shuffle=True)

            ##########################################
            # set up model and optimizer
            ##########################################
            net = RandomForestClassifier(n_estimators=model_width[o_ind], max_depth=10)
            t_acc, t_nums = 0, 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.reshape(-1, 28*28), targets
                net.fit(inputs, targets)
                y_pred = net.predict(inputs)
                acc_score = accuracy_score(targets, y_pred)
                t_acc += float(acc_score)
                t_nums += 1
            train_acc = t_acc / t_nums
            acc, nums = 0, 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.reshape(-1, 28*28), targets
                y_pred = net.predict(inputs)
                acc_score = accuracy_score(targets, y_pred)
                acc += float(acc_score)
                nums += 1
            test_acc = acc / nums
            # TRAIN_LOSS_SUM += train_loss
            # TEST_LOSS_SUM += test_loss
            TRAIN_ACC_SUM += train_acc
            TEST_ACC_SUM += test_acc
            log(logfilename, "{}\t{:.5}\t{:.5}".format(t, TRAIN_ACC_SUM / (t + 1), TEST_ACC_SUM / (t + 1)))
            print('trial: {}, train acc: {}, test acc: {}'.format(t, TRAIN_ACC_SUM / (t + 1), TEST_ACC_SUM / (t + 1)))
            current_train_metrics.append('trial: {}, train acc: {}, test acc: {}'.format(t, train_acc, test_acc))

print('Program finished', flush=True)
