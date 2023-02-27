#%%
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.resnet import ResNet18, M_ResNet18

random_seed = 113
pl.seed_everything(random_seed) # pytorch에서 random seed 넣는 네 줄을 알아서 처리해줌
#%%
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
mnist_train = MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

train_loader_mnist = DataLoader(
    mnist_train,
    batch_size=batch_size,
    shuffle=True,
)
#%%
class ResNetWrapperBasic(pl.LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.model = ResNet18()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, label = batch
        train_loss = self.common_step(batch, batch_idx)
        self.log("train_loss", train_loss) # ModelCheckpoint에서 moniter에 해당되는 string을 받을 수 있게함.
        # with torch.no_grad(): # with 안에 있는 코드가 contextmanager 때문에 추가적인 영향을 받음
        #     accuracy = ...
        #     self.log("train_accuracy", accuracy)
        return train_loss
        # ex) metrics = {"train_loss", [...]}
    # training step zero grad backward step이 loss를 epoch 당 코드를 짜서 for loop에 돌리도록 했는데,

    # lightning에서는 알아서 자동으로 채워줘서 loss 합까지 해줌

    def common_step(self, batch, batch_idx):
        image, label = batch
        label_pred = self.model(image)
        loss = F.nll_loss(label_pred, label)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=self.lr)
#%%
## Hack to work around the current torchvision bug involving libcudnn8.so
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

ckpt_best = ModelCheckpoint(monitor="train_loss", save_top_k = 1, mode="min") # 전체학습이 되는 전 구간 중에 loss가 가장 낮은 epoch에 해당하는 모델을 save하는 것. save top k model, based on the monitered metric
logger = WandbLogger(project="cifar10_test", entity="jhelab_team")
trainer = pl.Trainer(
    max_epochs = 3, # max_epochs을 안써놓으면 default가 있거나 계속 돔.
    accelerator = "cpu", # 학습하고자 하는 프로세스의 종류
    devices = 1, # multi 장치학습 -> 동시에 여러 개를 써서 학습하는 것(CPU, GPU, TPU etc..)
    auto_select_gpus = True, # GPU가 여러 장 있는 경우, 무슨 gpu를 사용할 건지에 대한 내용(무슨 GPU인지는 알아서 찾아감)
    deterministic = "warn", # True(같은결과), False(빠르게), warn 가급적 deterministic하게 하고 안되면 경고를 띄워줘라. warn을 놓는걸 권장. 
    callbacks = [ckpt_best],
)
#%%
lightning_model = ResNetWrapperBasic()
trainer.fit(lightning_model, train_dataloaders=mnist_train, val_dataloaders=mnist_train)

#%%
# with statement example
filepath = "myfile.txt"
with open(filepath, "r") as file:
    for line in file:
        ...

file = open(filepath, "r")
for line in file:
    ...
file.close()