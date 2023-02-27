#%%
## Library import, set environment variables

# Standard libraries
import os
import random
from typing import Type
import time

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# Custom libraries
from models.resnet import ResNet18, M_ResNet18
from models.convnet import Net, NewActivationNet
# .은 현재폴더를 의미 / __init__.py를 만들어야 인식을 할 수 있음

random_seed = 113
os.environ["PYTHONHASHSEED"] = str(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# cudnn -> libraries(nvdia, cuda랑 cudnn으로 학습)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True 
# 이건 알아야함. 수치연산의 값이 매번 정확하게 같이 나오는게 아니고 되게 작은 torelence안에서 값이 나오게 하면 조금 흔들릴 수 있는데, true라고 하면 아예 같은 결과를 나오게 할 수 있다.
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
# if device == "cuda":
#     torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

#%%
## Define dataset, make dataloaders
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),  # Mean, Std
    ]
)

trainset = CIFAR10(root="./data", train=True, download=False, transform=transform)
testset = CIFAR10(root="./data", train=False, download=False, transform=transform)

batch_size = 32
trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

testloader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)
#%%
## Inspect dataset before training
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

print(type(trainset), len(trainset), len(testset))
print(type(trainset[0]), len(trainset[0]))
print(len(trainset[0][0]))
print(len(trainset[0][0][0]))
print(len(trainset[0][0][0][0]))

image, label = trainset[0]
print(type(image), type(label))
print(image.shape, label)

class_items = trainset.classes
print(type(class_items), len(class_items))
print(class_items)
print(class_items[label])

plt.imshow(image.permute(1, 2, 0))
# %%
## Plot first batch from the train dataloader
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# functions to show an image
def imshow(img):
    img = img / 5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# image가 안나옴

# show images
imshow(make_grid(images))
# print labels
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))
#%%
# Define model
# m_net = M_ResNet18()
# net = Net()
# new_net = NewActivationNet()

net = ResNet18()

#%%
## Define optimizer
# **kwargs -> wild card argument 아무거나 들어갈 수 있음

learning_rate = 0.001
momentum = 0.9
def func(a, **kwargs):
    print(kwargs)
    return a
func(1.0, momentum = 0.8, acceleration = 0.8)

#%%
def make_optimizer(
        optimizer_class : Type[optim.Optimizer],
        model : nn.Module, 
        lr : float, 
        **kwargs
        )->optim.Optimizer:
    optimizer = optimizer_class(model.parameters(), lr=lr, **kwargs)
    return optimizer

optimizer = make_optimizer(optim.SGD, net, lr=learning_rate, momentum=momentum)
# m_optimizer = make_optimizer(optim.SGD, m_net, learning_rate, momentum=momentum)

#%%

# m_net = m_net.to(device)

#%%
# Train helper functions
def train_epoch(
        model : nn.Module, 
        optimizer : optim.Optimizer, 
        train_dataloader : DataLoader
        )->torch.Tensor:
    
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        # get the inputs; data is a list of [inputs, labels]
        image, label = batch
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad() # clear derivative information from previous loop

        # forward + backward + optimize
        label_pred = model(image) # obtain model prediction 
        loss_minibatch = F.nll_loss(label_pred, label) # calculate mimibatch loss
        loss_minibatch.backward() # signal end of computation graph, calculate derivatives
        optimizer.step() # update leaf tensors using update rule(sgd: x <- x-lr*D_x_L)
        
        # add minibath contribution to the total epoch loss
        train_loss += loss_minibatch.item()
    return train_loss


def test_epoch(model, optimizer, dataloader):
    net.eval()
    test_loss = 0
    correct = 0
    # correct_1 = 0
    # correct_5 = 0
    with torch.no_grad():
        for input, label in testloader:
            input, label = input.to(device), label.to(device)
            output = net(input)
            test_loss += F.nll_loss(output, label, size_average=False).item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(label.data.view_as(prediction)).sum()
            for label, prediction in zip(label, prediction):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)
    corrects.append(correct)
    print(
        f"Test set: Avg. loss: {test_loss:.4f}, Accuracy: {100.0 * correct / len(testloader.dataset):.2f}%"
    )
#%%
## Run training loop
max_epochs = 3
logger_interval = 1
start_time = time.time()

net = net.to(device)
train_losses = []
# test_epoch()
for epoch in range(max_epochs):
    train_loss = train_epoch(net, optimizer, trainloader)
    train_losses.append(train_loss)
    if epoch % logger_interval == 0:
        print(
            f"Train Epoch:{epoch}, loss: {train_loss:.4f}"
        )
        # torch.save(net.state_dict(), "./results/cifar_net.pth")
        # torch.save(optimizer.state_dict(), "./results/cifar_net.pth")

    test_epoch()  # loop over the dataset multiple times

total_time = time.time() - start_time
print(f" Duration: {total_time/60} mins")

#%%
train_counter = []
test_losses = []
test_counter = [i * len(trainloader.dataset) for i in range(max_epochs + 1)]
corrects = []
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


m_train_losses = []
m_train_counter = []
m_test_losses = []
m_test_counter = [i * len(trainloader.dataset) for i in range(max_epochs + 1)]
m_corrects = []




#%%
def m_train(epoch):
    m_net.train()
    m_train_loss = 0
    for batch_idx, (m_input, label) in enumerate(trainloader):
        m_input, label = m_input.to(device), label.to(device)

        m_optimizer.zero_grad()

        # forward + backward + optimize
        m_output = m_net(m_input)
        m_loss = F.nll_loss(m_output, label)
        m_loss.backward()
        m_optimizer.step()

        # print statistics
        m_train_loss += m_loss.item()
        if batch_idx % batch_size == batch_size - 1:  # print every mini-batches
            m_train_losses.append(m_train_loss.item())
            print(
                f"Train Epoch:{epoch}, batch index:{batch_idx + 1}, loss: {m_train_loss / 2000:.4f}"
            )
            torch.save(m_net.state_dict(), "./results/cifar_m_net.pth")
            torch.save(m_optimizer.state_dict(), "./results/cifar_m_net.pth")


def m_test():
    m_net.eval()
    m_test_loss = 0
    m_correct = 0
    with torch.no_grad():
        for m_input, label in testloader:
            m_input, label = m_input.to(device), label.to(device)
            m_output = m_net(m_input)
            m_test_loss += F.nll_loss(m_output, label, size_average=False).item()
            m_pred = m_output.data.max(1, keepdim=True)[1]
            m_correct += m_pred.eq(label.data.view_as(m_pred)).sum()
    m_test_loss /= len(testloader.dataset)
    m_test_losses.append(m_test_loss)
    m_corrects.append(m_correct)
    print(
        f"Test set: Avg. loss: {m_test_loss:.4f}, Accuracy: {100.0 * m_correct / len(testloader.dataset):.2f}%"
    )


#%%

# batch 32
# 3 max_epochs
# Test set: Avg. loss: 0.9021, Accuracy: 68.01%
# Test set: Avg. loss: 0.7579, Accuracy: 73.94%
# Test set: Avg. loss: 0.6791, Accuracy: 76.80%

# 10 max_epochs
# Test set: Avg. loss: 0.9021, Accuracy: 68.01%
# Test set: Avg. loss: 0.7579, Accuracy: 73.94%
# Test set: Avg. loss: 0.6791, Accuracy: 76.80%
# Test set: Avg. loss: 0.6834, Accuracy: 77.64%
# Test set: Avg. loss: 0.7301, Accuracy: 78.01%
# Test set: Avg. loss: 0.7879, Accuracy: 77.80%
# Test set: Avg. loss: 0.7952, Accuracy: 78.41%
# Test set: Avg. loss: 0.8182, Accuracy: 79.04%
# Test set: Avg. loss: 0.9138, Accuracy: 78.25%
# Test set: Avg. loss: 0.7859, Accuracy: 81.13%

# batch 64
# 3 max_epochs
# Test set: Avg. loss: 1.0751, Accuracy: 61.89%
# Test set: Avg. loss: 0.8413, Accuracy: 70.10%
# Test set: Avg. loss: 0.7413, Accuracy: 74.27%
#%%
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.2f} %")
#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.bar(correct_pred.keys(), correct_pred.values())
#%%
correct_pred.keys(), correct_pred.items()
#%%
total_pred
#%%
from more_itertools import take

top_5 = dict(take(5, correct_pred.items()))
top_5
#%%
sorted_dicts = dict(
    sorted(correct_pred.items(), key=lambda item: item[1], reverse=True)
)
sorted_dicts
#%%
top_5 = dict(take(5, sorted_dicts.items()))
top_5
#%%
top_5_accuracy_dicts = {key: value / 4000 for key, value in top_5.items()}
top_5_accuracy_dicts


#%%
import time

start_time = time.time()
m_test()
for epoch in range(1, max_epochs + 1):
    m_train(epoch)
    m_test()

total_time = time.time() - start_time
print(f" Duration: {total_time/60} mins")

# 32 batch
# 4 max_epochs
# 1, 1
# Test set: Avg. loss: 0.8627, Accuracy: 68.98%
# Test set: Avg. loss: 0.7092, Accuracy: 75.53%
# Test set: Avg. loss: 0.6331, Accuracy: 78.26%
# Test set: Avg. loss: 0.7129, Accuracy: 77.30%

# 10 max_epochs
# Test set: Avg. loss: 0.8627, Accuracy: 68.98%
# Test set: Avg. loss: 0.7092, Accuracy: 75.53%
# Test set: Avg. loss: 0.6331, Accuracy: 78.26%
# Test set: Avg. loss: 0.7129, Accuracy: 77.30%
# Test set: Avg. loss: 0.6951, Accuracy: 79.29%
# Test set: Avg. loss: 0.7547, Accuracy: 79.04%
# Test set: Avg. loss: 0.7569, Accuracy: 79.83%
# Test set: Avg. loss: 0.8708, Accuracy: 78.88%
# Test set: Avg. loss: 0.8648, Accuracy: 79.84%
# Test set: Avg. loss: 0.7960, Accuracy: 80.97%
#%%
from torchinfo import summary

examples = enumerate(testloader)
batch_idx, (example_data, example_labels) = next(examples)

example_data.shape
#%%
summary(net, (batch_size, 3, 32, 32))
#%%
batch_idx = iter(testloader)
images, labels = next(batch_idx)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# print images
img_grid = make_grid(images)
imshow(img_grid)
print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(32)))

#%%
#
#
#
#
#
# Define a loss function and optimizer
# import torch.nn.functional as F
# from models.resnet import ResNet18
# import torch.optim as optim

# learning_rate = 0.001
# momentum = 0.9

# net = ResNet18()
# net = net.to(device)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# # Train the network
# for epoch in range(5):  # loop over the dataset multiple times

#     train_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data.to(device), labels.to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = F.nll_loss(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         train_loss += loss.item()
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print(f"[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 2000:.4f}")
#             train_loss = 0.0

# print("Finished Training")
# PATH = "./results/cifar_new_net.pth"
# torch.save(net.state_dict(), PATH)
# %%
#
#
#
#
# #
# net = ResNet18()
# net = net.to(device)
# net.load_state_dict(torch.load(PATH))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)

# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(
#     f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
# )

# # 3 max_epochs
# # 2, 2, Accuracy of the network on the 10000 test images: 61 %

# %%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data, label in testloader:
        images, labels = data.to(device), label.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.2f} %")
# 32 batch
# 4 max_epochs
# Accuracy for class: plane is 84.40 %
# Accuracy for class: car   is 89.50 %
# Accuracy for class: bird  is 76.50 %
# Accuracy for class: cat   is 62.70 %
# Accuracy for class: deer  is 80.40 %
# Accuracy for class: dog   is 47.40 %
# Accuracy for class: frog  is 83.10 %
# Accuracy for class: horse is 86.50 %
# Accuracy for class: ship  is 81.40 %
# Accuracy for class: truck is 89.10 %

# 64 batch
# 3 max_epochs
# Accuracy for class: plane is 81.60 %
# Accuracy for class: car   is 89.60 %
# Accuracy for class: bird  is 59.20 %
# Accuracy for class: cat   is 45.80 %
# Accuracy for class: deer  is 71.60 %
# Accuracy for class: dog   is 71.70 %
# Accuracy for class: frog  is 79.20 %
# Accuracy for class: horse is 83.30 %
# Accuracy for class: ship  is 77.50 %
# Accuracy for class: truck is 83.20 %

# %%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data, label in testloader:
        images, labels = data.to(device), label.to(device)
        outputs = m_net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.2f} %")

# 4 max_epochs
# Accuracy for class: plane is 83.10 %
# Accuracy for class: car   is 94.20 %
# Accuracy for class: bird  is 61.90 %
# Accuracy for class: cat   is 43.90 %
# Accuracy for class: deer  is 71.70 %
# Accuracy for class: dog   is 76.20 %
# Accuracy for class: frog  is 85.70 %
# Accuracy for class: horse is 81.10 %
# Accuracy for class: ship  is 87.40 %
# Accuracy for class: truck is 87.80 %

# 10 max_epochs
# Accuracy for class: plane is 85.20 %
# Accuracy for class: car   is 94.00 %
# Accuracy for class: bird  is 73.40 %
# Accuracy for class: cat   is 68.00 %
# Accuracy for class: deer  is 71.30 %
# Accuracy for class: dog   is 73.00 %
# Accuracy for class: frog  is 82.80 %
# Accuracy for class: horse is 86.20 %
# Accuracy for class: ship  is 90.70 %
# Accuracy for class: truck is 85.10 %

# %%
img = images[0].view(1, 3072)

# we are turning off the gradients
with torch.no_grad():
    model_prediction = net.forward(img)
probabilities = F.softmax(model_prediction, dim=1).detach().cpu().numpy().squeeze()

print(probabilities)
#%%
img_grid = make_grid(images)
imshow(img_grid)
net = ResNet18()
net.load_state_dict(torch.load(PATH))
fig, (ax1, ax2) = plt.subplots(figsize=(6, 8), ncols=2)
img = img.view(3, 32, 32)

ax1.axis("off")
ax2.barh(np.arange(10), probabilities, color="r")
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(**classes)
ax2.set_title("Class Probability")
ax2.set_xlim(0, 1.1)

plt.tight_layout()

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(range(4), test_losses)
# %%
train_losses
# %%
test_losses
# %%
