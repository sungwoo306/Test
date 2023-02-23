#%%
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from models.resnet import ResNet18, M_ResNet18

random_seed = 113
pl.seed_everything(random_seed)
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
class ResNetWrapper(pl.Lightningmodule):
    def __init__(self, lr=0.01):
        self.model = ResNet18()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        train_loss = self.common_step(batch, batch_idx)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.common_step(batch, batch_idx)
        self.log("test_loss", test_loss)
        return test_loss

    def common_step(self, batch, batch_idx):
        img, label = batch
        label_pred = self.model(img)
        loss = F.nll_loss(label_pred, label)
        return loss

    def configure_optimizers(self):
        return SGD(self.model.parameters(), lr=self.lr)


lightning_model = ResNetWrapper()
logger = WandbLogger(project=None, entity=None)
trainer = pl.Trainer(callbacks=[], logger=logger)
#%%
trainer.fit(lightning_model, train_dataloaders=trainloader, val_dataloaders=valloader)