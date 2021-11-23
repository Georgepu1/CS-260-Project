# Source: https://colab.research.google.com/drive/1MTk5fHJf3eG5LAE2VlvHM-h49f36DAVd?usp=sharing#scrollTo=amcERPImwUa9
# Source: https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff/blob/5fc6ab3a416ecb96759b25958a379f620d07fbf9/mnist/train.py
# Source: https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff/blob/5fc6ab3a416ecb96759b25958a379f620d07fbf9/cifar10/train_labelnoise.py

import os
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from random import shuffle
import copy

# Standard NN definition
class DNN(nn.Module):
    def __init__(self, width=1):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(784, width, bias=False)
        self.fc2 = nn.Linear(width, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(torch.flatten(x, 1))
        out = self.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out

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


# Methods for testing NN
# Training
def train(net, trainloader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets_onehot)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * outputs.numel()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss / total, 100. * correct / total


# Test
def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
            targets_onehot.zero_()
            targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
            outputs = net(inputs)
            loss = criterion(outputs, targets_onehot)
            test_loss += loss.item() * outputs.numel()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return test_loss / total, 100. * correct / total


def compute_bias_variance(net, testloader, trial):
    net.eval()
    bias2 = 0
    variance = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
            targets_onehot.zero_()
            targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
            outputs = net(inputs)
            OUTPUST_SUM[total:(total + targets.size(0)), :] += outputs
            OUTPUTS_SUMNORMSQUARED[total:total + targets.size(0)] += outputs.norm(dim=1) ** 2.0

            bias2 += (OUTPUST_SUM[total:total + targets.size(0), :] / (trial + 1) - targets_onehot).norm() ** 2.0
            variance += OUTPUTS_SUMNORMSQUARED[total:total + targets.size(0)].sum()/(trial + 1) - (OUTPUST_SUM[total:total + targets.size(0), :]/(trial + 1)).norm() ** 2.0
            total += targets.size(0)

    return bias2 / total, variance / total

dataset = 'mnist'
outdir=''
trial=5
weight_decay=5e-4
lr=0.1
lr_decay=100
num_epoch=200
test_size=10000
save_freq=100

# loss definition
criterion = nn.MSELoss(reduction='mean').cuda()
# variables for bias-variance calculation
NUM_CLASSES = 10
OUTPUST_SUM = torch.Tensor(test_size, NUM_CLASSES).zero_().cuda()
OUTPUTS_SUMNORMSQUARED = torch.Tensor(test_size).zero_().cuda()

# Track experiments
current_metrics = []
current_train_metrics = []

ratio = [0.0, 0.2, 0.4, 0.6]
model_width = [50]
# set up index after random permutation
permute_index = np.split(np.random.permutation(len(trainset)), trial)

for o_ind in range(len(model_width)):
  for ind in range(len(ratio)):
    TRAIN_ACC_SUM = 0.0
    TEST_ACC_SUM = 0.0
    TRAIN_LOSS_SUM = 0.0
    TEST_LOSS_SUM = 0.0
    outdir = ''
    outdir = '{}_trial{}_mse{}_noise{}'.format(dataset, trial, outdir, ratio[ind])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logfilename = os.path.join(outdir, 'log_width{}.txt'.format(model_width[o_ind]))
    init_logfile(logfilename, "trial\ttrain_loss\ttrain acc\ttest loss\ttest acc\tbias2\tvariance")
    current_metrics.append('--Break--')
    for t in range(trial):
      ##########################################
      # set up subsampled train loader
      ##########################################
      # trainsubset = get_subsample_dataset(trainset, permute_index[t])
      trainsubset = get_subsample_dataset_label_noise(trainset, permute_index[t], int(len(permute_index[t])*ratio[ind]))
      trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=128, shuffle=True)

      ##########################################
      # set up model and optimizer
      ##########################################
      net = DNN(width=model_width[o_ind]).cuda()
      optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
      scheduler = StepLR(optimizer, step_size=lr_decay, gamma=0.1)

      for epoch in range(1, num_epoch + 1):
          train_loss, train_acc = train(net, trainloader)
          test_loss, test_acc = test(net, testloader)
          print('epoch: {}, train_loss: {:.6f}, train acc: {}, test loss: {:.6f}, test acc: {}'.format(epoch, train_loss, train_acc, test_loss, test_acc))
          current_train_metrics.append('epoch: {}, train_loss: {:.6f}, train acc: {}, test loss: {:.6f}, test acc: {}'.format(epoch, train_loss, train_acc, test_loss, test_acc))
          scheduler.step()
          if epoch % save_freq == 0:
              torch.save(net.state_dict(), os.path.join(outdir, 'model_width{}_trial{}_epoch{}_noise{}.pkl'.format(model_width[o_ind], t, epoch, ratio[ind])))

      TRAIN_LOSS_SUM += train_loss
      TEST_LOSS_SUM += test_loss
      TRAIN_ACC_SUM += train_acc
      TEST_ACC_SUM += test_acc

      # compute bias and variance
      bias2, variance = compute_bias_variance(net, testloader, t)
      variance_unbias = variance * t / (t - 1.0)
      bias2_unbias = TEST_LOSS_SUM / (t + 1) - variance_unbias
      print('trial: {}, train_loss: {:.6f}, train acc: {}, test loss: {:.6f}, test acc: {}, bias2: {}, variance: {}'.format(
          t, TRAIN_LOSS_SUM / (t + 1), TRAIN_ACC_SUM / (t + 1), TEST_LOSS_SUM / (t + 1),
          TEST_ACC_SUM / (t + 1), bias2_unbias, variance_unbias))
      log(logfilename, "{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}".format(
          t, TRAIN_LOSS_SUM / (t + 1), TRAIN_ACC_SUM / (t + 1), TEST_LOSS_SUM / (t + 1),
          TEST_ACC_SUM / (t + 1), bias2_unbias, variance_unbias))

      current_train_metrics.append('trial: {}, train_loss: {:.6f}, train acc: {}, test loss: {:.6f}, test acc: {}, bias2: {}, variance: {}'.format(
          t, TRAIN_LOSS_SUM / (t + 1), TRAIN_ACC_SUM / (t + 1), TEST_LOSS_SUM / (t + 1),
          TEST_ACC_SUM / (t + 1), bias2_unbias, variance_unbias))

      current_metrics.append((bias2, variance, variance_unbias, bias2_unbias, TRAIN_LOSS_SUM / (t + 1), TRAIN_ACC_SUM / (t + 1), TEST_LOSS_SUM / (t + 1), TEST_ACC_SUM / (t + 1)))
      # save the model
      torch.save(net.state_dict(), os.path.join(outdir, 'model_width{}_trial{}_noise{}.pkl'.format(model_width[o_ind], t, ratio[ind])))

print('Program finished', flush=True)

# If want to save
import pickle as pkl
pkl.dump(current_metrics, open('output-50.pkl', 'wb'))
pkl.dump(current_train_metrics, open('output-train-50.pkl', 'wb'))

# If want to load
# with open('output.pkl','rb') as f:
#   x = pkl.load(f)

# Mount driver to authenticate yourself to gdrive
# from google.colab import drive
# drive.mount('/content/gdrive')

# if want to save to gdrive
# import numpy as np
# from numpy import savetxt

# np_metrics = np.array(current_metrics)
# np_train_metrics = np.array(current_train_metrics)

# with open('/content/gdrive/My Drive/output-50.txt', 'w') as f:
#     np.savetxt(f, np_metrics, delimiter=" ", fmt="%s")

# with open('/content/gdrive/My Drive/output-train-50.txt', 'w') as f:
#     np.savetxt(f, np_train_metrics, delimiter=" ", fmt="%s")

