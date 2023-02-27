#%%
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random


batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10
random_seed = 113

random.seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(random_seed)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.cuda.manual_seed(random_seed)
# if device == "cuda":
#     torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


mnist_trainsets = datasets.MNIST(
    root="./data", train=True, download=True, transform=None
)
mnist_testset = datasets.MNIST(
    root="./data", train=False, download=True, transform=None
)
train_loader = DataLoader(
    datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_train,
    shuffle=True,
)

test_loader = DataLoader(
    datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=True,
)
#%%
len(mnist_trainsets), len(mnist_testset)
#%%
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape
#%%
import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
# %%
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from activations.activations import molu


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input size = 28x28
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=0)
        # ((W-K+2P)/S)+1,  ((28-5+0)/1)+1=24 -> 24x24
        # maxpooling -> 12x12
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=0)
        # ((12-5+0)/1)+1=8 -> 8x8
        # maxpooling -> 4x4 : To prevent from overfitting
        self.conv2_drop = nn.Dropout2d(p=0.2, inplace=False)
        # To help promote independence between feature maps
        self.fc1 = nn.Linear(320, 50)
        # Flatten 4x4x20, then transform into 100
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


net = Net()
net = net
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


class NewActivationNet(nn.Module):
    def __init__(self):
        super(NewActivationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=0)
        self.conv2_drop = nn.Dropout2d(p=0.2, inplace=False)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = molu(F.max_pool2d(self.conv1(x), 2))
        x = molu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = molu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


new_net = NewActivationNet()
new_net = new_net
new_optimizer = optim.SGD(
    new_net.parameters(), lr=learning_rate, momentum=momentum
)
#%%

n_epochs = 3


train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
corrects = []

new_train_losses = []
new_train_counter = []
new_test_losses = []
new_test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
new_corrects = []


def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
            torch.save(net.state_dict(), "./results/model.pth")
            torch.save(optimizer.state_dict(), "./results/optimizer.pth")


def new_train(epoch):
    new_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        new_optimizer.zero_grad()
        new_output = new_net(data)
        new_loss = F.nll_loss(new_output, target)
        new_loss.backward()
        new_optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    new_loss.item(),
                )
            )
            new_train_losses.append(new_loss.item())
            new_train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
            torch.save(new_net.state_dict(), "./results/new_model.pth")
            torch.save(
                new_optimizer.state_dict(), "./results/new_optimizer.pth"
            )


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    corrects.append(correct)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def new_test():
    new_net.eval()
    new_test_loss = 0
    new_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            new_output = new_net(data)
            new_test_loss += F.nll_loss(
                new_output, target, size_average=False
            ).item()
            new_pred = new_output.data.max(1, keepdim=True)[1]
            new_correct += new_pred.eq(target.data.view_as(new_pred)).sum()
    new_test_loss /= len(test_loader.dataset)
    new_test_losses.append(new_test_loss)
    new_corrects.append(new_correct)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            new_test_loss,
            new_correct,
            len(test_loader.dataset),
            100.0 * new_correct / len(test_loader.dataset),
        )
    )


# %%
# import time

# start_time = time.time()

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# total_time = time.time() - start_time
# print(f" Duration: {total_time} sec")

# 3 epochs 22.173375606536865 sec
# Test set: Avg. loss: 0.3547, Accuracy: 9065/10000 (90.65%)

# cuda
# 3 epochs
# Test set: Avg. loss: 0.3520, Accuracy: 9048/10000 (90.48%)

# lr=0.01
# Test set: Avg. loss: 0.0725, Accuracy: 9762/10000 (97.62%)

# elu
# 3 epochs
# Test set: Avg. loss: 0.2820, Accuracy: 9216/10000 (92.16%)

# tanh
# 3 epochs
# Test set: Avg. loss: 0.7366, Accuracy: 8396/10000 (83.96%)
#%%
# import time

# start_time = time.time()

new_test()
for epoch in range(1, n_epochs + 1):
    new_train(epoch)
    new_test()

# total_time = time.time() - start_time
# print(f" Duration: {total_time} sec")

# 3 epochs
# 1, 1, Test set: Avg. loss: 0.2875, Accuracy: 9232/10000 (92.32%)
# 1, 2, Test set: Avg. loss: 0.3010, Accuracy: 9210/10000 (92.10%)
# 1, 0.5, Test set: Avg. loss: 0.2814, Accuracy: 9227/10000 (92.27%)
# 0.5, 0.5, Test set: Avg. loss: 0.3464, Accuracy: 9090/10000 (90.90%)
# 0.7, 1.3, Test set: Avg. loss: 0.3009, Accuracy: 9206/10000 (92.06%)
# 0.8, 1.3, Test set: Avg. loss: 0.2972, Accuracy: 9220/10000 (92.20%)
# 0.5, 2, Test set: Avg. loss: 0.3227, Accuracy: 9171/10000 (91.71%)
# 0.9, 1, Test set: Avg. loss: 0.2904, Accuracy: 9227/10000 (92.27%)
# 1, 1.1, Test set: Avg. loss: 0.2888, Accuracy: 9226/10000 (92.26%)
# 1.1, 1.1, Test set: Avg. loss: 0.2863, Accuracy: 9232/10000 (92.32%)
# 1.2, 1.2, Test set: Avg. loss: 0.2854, Accuracy: 9235/10000 (92.35%)
# 1.5, 1.5, Test set: Avg. loss: 0.2839, Accuracy: 9250/10000 (92.50%)
# 2, 2, Test set: Avg. loss: 0.2837, Accuracy: 9253/10000 (92.53%)
# 3, 3, Test set: Avg. loss: 0.2860, Accuracy: 9240/10000 (92.40%)
# 2.5, 2.5, Test set: Avg. loss: 0.2846, Accuracy: 9247/10000 (92.47%)

# 3 epochs
# 1, 1, Test set: Avg. loss: 0.2895, Accuracy: 9240/10000 (92.40%)
# 2, 2, Test set: Avg. loss: 0.2855, Accuracy: 9263/10000 (92.63%)

# lr=0.01
# 1, 1, Test set: Avg. loss: 0.0634, Accuracy: 9801/10000 (98.01%)
# %%
with torch.no_grad():
    output = net(example_data)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
    plt.title(
        "Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item())
    )
    plt.xticks([])
    plt.yticks([])
# plt.savefig("figures/relu-1.png")
# %%
with torch.no_grad():
    new_output = new_net(example_data)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
    plt.title(
        "Prediction: {}".format(
            new_output.data.max(1, keepdim=True)[1][i].item()
        )
    )
    plt.xticks([])
    plt.yticks([])
# plt.savefig("figures/e,e-1.png")
# %%
continued_network = Net()
continued_optimizer = optim.SGD(
    net.parameters(), lr=learning_rate, momentum=momentum
)

network_state_dict = torch.load("./results/model.pth")
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load("./results/optimizer.pth")
continued_optimizer.load_state_dict(optimizer_state_dict)

for i in range(4, 11):
    test_counter.append(i * len(train_loader.dataset))
    train(i)
    test()

# 10 epochs 70.99121451377869 sec
# Test set: Avg. loss: 0.1409, Accuracy: 9590/10000 (95.90%)
# elu : Test set: Avg. loss: 0.1160, Accuracy: 9639/10000 (96.39%)
# tanh : Test set: Avg. loss: 0.2469, Accuracy: 9332/10000 (93.32%)

# 20 epochs 140.57183241844177 sec
# Test set: Avg. loss: 0.0825, Accuracy: 9752/10000 (97.52%)
# elu : Test set: Avg. loss: 0.0734, Accuracy: 9782/10000 (97.82%)
# tanh : Test set: Avg. loss: 0.1436, Accuracy: 9578/10000 (95.78%)

# 30 epochs 209.6886875629425 sec
# Test set: Avg. loss: 0.0725, Accuracy: 9779/10000 (97.79%)
# elu : Test set: Avg. loss: 0.0575, Accuracy: 9828/10000 (98.28%)
# tanh : Test set: Avg. loss: 0.1057, Accuracy: 9662/10000 (96.62%)

# 40 epochs 293.6556088924408 sec
# Test set: Avg. loss: 0.0560, Accuracy: 9837/10000 (98.38%)
# Test set: Avg. loss: 0.0546, Accuracy: 9841/10000 (98.41%)
# Test set: Avg. loss: 0.0546, Accuracy: 9840/10000 (98.40%)
# elu : Test set: Avg. loss: 0.0495, Accuracy: 9847/10000 (98.47%)
# tanh : Test set: Avg. loss: 0.0854, Accuracy: 9728/10000 (97.28%)

# 50 epochs 347.1954593658447 sec
#
# Test set: Avg. loss: 0.0492, Accuracy: 9862/10000 (98.62%)
# Test set: Avg. loss: 0.0493, Accuracy: 9863/10000 (98.63%)
# elu : Test set: Avg. loss: 0.0442, Accuracy: 9867/10000 (98.67%)
# tanh : Test set: Avg. loss: 0.0728, Accuracy: 9770/10000 (97.70%)
# %%
continued_new_network = NewActivationNet()
continued_new_optimizer = optim.SGD(
    new_net.parameters(), lr=learning_rate, momentum=momentum
)

new_network_state_dict = torch.load("./results/new_model.pth")
continued_new_network.load_state_dict(new_network_state_dict)

new_optimizer_state_dict = torch.load("./results/new_optimizer.pth")
continued_new_optimizer.load_state_dict(new_optimizer_state_dict)

for i in range(4, 11):
    new_test_counter.append(i * len(train_loader.dataset))
    new_train(i)
    new_test()

# 10 epochs
# 1, 1, Test set: Avg. loss: 0.1190, Accuracy: 9640/10000 (96.40%)
# 1, 2, Test set: Avg. loss: 0.1229, Accuracy: 9614/10000 (96.14%)
# 1, 0.5, Test set: Avg. loss: 0.1166, Accuracy: 9647/10000 (96.47%)
# 2, 2, Test set: Avg. loss: 0.1179, Accuracy: 9635/10000 (96.35%)

# 1, 1, Test set: Avg. loss: 0.1200, Accuracy: 9638/10000 (96.38%)
# 2, 2, Test set: Avg. loss: 0.1186, Accuracy: 9643/10000 (96.43%)

# 20 epochs
# 1, 1, Test set: Avg. loss: 0.0758, Accuracy: 9763/10000 (97.63%)
# 1, 1, Test set: Avg. loss: 0.0768, Accuracy: 9770/10000 (97.70%)
# 2, 2, Test set: Avg. loss: 0.0757, Accuracy: 9771/10000 (97.71%)

# 30 epochs
# 1, 1, Test set: Avg. loss: 0.0597, Accuracy: 9814/10000 (98.14%)
# 1, 1, Test set: Avg. loss: 0.0602, Accuracy: 9812/10000 (98.12%)
# 2, 2, Test set: Avg. loss: 0.0598, Accuracy: 9815/10000 (98.15%)

# 40 epochs
#
# 1, 1, Test set: Avg. loss: 0.0507, Accuracy: 9827/10000 (98.27%)
# 1, 1, Test set: Avg. loss: 0.0508, Accuracy: 9830/10000 (98.30%)
# 2, 2, Test set: Avg. loss: 0.0511, Accuracy: 9832/10000 (98.32%)

# 50 epochs
#
# 1, 1, Test set: Avg. loss: 0.0441, Accuracy: 9855/10000 (98.55%)
# 1, 1, Test set: Avg. loss: 0.0444, Accuracy: 9860/10000 (98.60%)
# 2, 2, Test set: Avg. loss: 0.0453, Accuracy: 9850/10000 (98.50%)

# lr=0.01, 1, 1
# 10, Test set: Avg. loss: 0.0360, Accuracy: 9885/10000 (98.85%)
# 20, Test set: Avg. loss: 0.0283, Accuracy: 9910/10000 (99.10%)
# 30, Test set: Avg. loss: 0.0254, Accuracy: 9923/10000 (99.23%)
# 40, Test set: Avg. loss: 0.0266, Accuracy: 9924/10000 (99.24%)
# 50, Test set: Avg. loss: 0.0275, Accuracy: 9914/10000 (99.14%)
# %%
len(train_losses), len(test_losses)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(range(940), train_losses, label="ReLU")
ax.plot(range(940), new_train_losses, label="MoLU")
ax.set_xlabel("idk")
ax.set_ylabel("Average of loss")
ax.grid(color="lightgray")
ax.legend(fontsize=13)
# fig.savefig("figures/1,1,loss.pdf", dpi=300, bbox_inches="tight")
# %%
import matplotlib.pyplot as plt

crs = []
new_crs = []
for i in range(51):
    c = corrects[i].numpy() / 100
    new_c = new_corrects[i].numpy() / 100
    crs.append(c)
    new_crs.append(new_c)

fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
ax.plot(range(51), crs, ".-", label="ReLU")
ax.plot(range(51), new_crs, ".-", label="MoLU")
ax.set_xlabel("Epochs", fontsize=11)
ax.set_ylabel("Average of accuracy(%)", fontsize=11)
ax.grid(color="lightgray")
ax.legend(fontsize=11, loc="center right")
fig.savefig("figures/1,1,acc5_2.pdf", dpi=300, bbox_inches="tight")
# %%
# Negative log likelihood loss
fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
ax.plot(train_counter, train_losses, label="Train loss of ReLU")
ax.plot(new_train_counter, new_train_losses, label="Train loss of MoLU")
ax.plot(test_counter, test_losses, ".-", label="Test loss of ReLU")
ax.plot(new_test_counter, new_test_losses, ".-", label="Test loss of MoLU")

ax.set_xlabel("Number of training examples seen", fontsize=13)
ax.set_ylabel("Average of loss", fontsize=13)
ax.grid(color="lightgray")
ax.legend(fontsize=13, loc="center right")
fig.savefig("figures/1,1,loss5_2.pdf", dpi=300, bbox_inches="tight")
# %%
