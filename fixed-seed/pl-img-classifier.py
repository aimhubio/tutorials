import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision import transforms
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger

class ImageClassifierModel(pl.LightningModule):

    def __init__(self, seed, lr, optimizer):
        super(ImageClassifierModel, self).__init__()
        pl.seed_everything(seed)
        # This fixes a seed for all the modules used by pytorch-lightning
        # Note: using random.seed(seed) is not enough, there are multiple
        # other seeds like hash seed, seed for numpy etc.
        self.lr = lr
        self.optimizer = optimizer
        self.total_classified = 0
        self.correctly_classified = 0
        self.model = resnet18(pretrained = True, progress = True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        # changing the last layer from 1000 out_features to 10 because the model
        # is pretrained on ImageNet which has 1000 classes but CIFAR10 has 10
        self.model.to('cuda') # moving the model to cuda

    def train_dataloader(self):
        # makes the training dataloader
        train_ds = CIFAR10('.', train = True, transform = transforms.ToTensor(), download = True)
        train_loader = DataLoader(train_ds, batch_size=32)
        return train_loader
        
    def val_dataloader(self):
        # makes the validation dataloader
        val_ds = CIFAR10('.', train = False, transform = transforms.ToTensor(), download = True)
        val_loader = DataLoader(val_ds, batch_size=32)
        return val_loader

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        # calculating the cross entropy loss on the result
        self.log('train_loss', loss)
        # logging the loss with "train_" prefix
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        # calculating the cross entropy loss on the result
        self.total_classified += y.shape[0]
        self.correctly_classified += (y_hat.argmax(1) == y).sum().item()
        # Calculating total and correctly classified images to determine the accuracy later
        self.log('val_loss', loss) 
        # logging the loss with "val_" prefix
        return loss

    def validation_epoch_end(self, results):
        accuracy = self.correctly_classified / self.total_classified
        self.log('val_accuracy', accuracy)
        # logging accuracy
        self.total_classified = 0
        self.correctly_classified = 0
        return accuracy

    def configure_optimizers(self):
        # Choose an optimizer and set up a learning rate according to hyperparameters
        if self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

if __name__ == "__main__":

    seeds = [47, 881, 123456789]
    lrs = [0.1, 0.01]
    optimizers = ['SGD', 'Adam']

    for seed in seeds:
        for optimizer in optimizers:
            for lr in lrs:
                # choosing one set of hyperparaameters from the ones above
            
                model = ImageClassifierModel(seed, lr, optimizer)
                # initializing the model we will train with the chosen hyperparameters

                aim_logger = AimLogger(
                    experiment='resnet18_classification',
                    train_metric_prefix='train_',
                    val_metric_prefix='val_',
                )
                aim_logger.log_hyperparams({
                    'lr': lr,
                    'optimizer': optimizer,
                    'seed': seed
                })
                # initializing the aim logger and logging the hyperparameters

                trainer = pl.Trainer(
                    logger=aim_logger,
                    gpus=1,
                    max_epochs=5,
                    progress_bar_refresh_rate=1,
                    log_every_n_steps=10,
                    check_val_every_n_epoch=1)
                # making the pytorch-lightning trainer

                trainer.fit(model)
                # training the model
