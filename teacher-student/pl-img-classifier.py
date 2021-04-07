import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
from torchvision.models import resnet18
from torchvision import transforms
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger

class ImageClassifierModel(pl.LightningModule):

    def __init__(self):
        super(ImageClassifierModel, self).__init__()
        self.total_classified = 0
        self.current_model = 'teacher'
        # keeping track whether we are on a stage of training the teacher or not
        self.correctly_classified = 0
        self.teacher_model = resnet18(pretrained = True, progress = True)
        self.student_model = resnet18(pretrained = True, progress = True)
        self.teacher_model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        self.student_model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        self.teacher_model.to('cuda')
        self.student_model.to('cuda')
        # changing the last layer from 1000 out_features to 10 because the model
        # is pretrained on ImageNet which has 1000 classes but STL10 has 10
        self.make_trainers()
        # self.model.to('cuda') # moving the model to cuda

    def make_trainers(self):
        self.aim_logger = AimLogger(
            experiment='resnet18_classification',
            train_metric_prefix='train_',
            val_metric_prefix='val_',
        )
        self.teacher_trainer = pl.Trainer(
            logger=self.aim_logger,
            gpus=1,
            max_epochs=50,
            progress_bar_refresh_rate=1,
            log_every_n_steps=10,
            check_val_every_n_epoch=1)
        self.student_trainer = pl.Trainer(
            logger=self.aim_logger,
            gpus=1,
            max_epochs=50,
            progress_bar_refresh_rate=1,
            log_every_n_steps=10,
            check_val_every_n_epoch=1)

    def train_dataloader(self):
        # makes the training dataloaders
        train_labeled_ds = STL10('.', split = 'train',
                                 transform = transforms.ToTensor(), download = True)
        train_labeled_ds = Subset(train_labeled_ds, torch.arange(500))
        train_labeled_loader = DataLoader(train_labeled_ds, batch_size=64)
        train_unlabeled_ds = STL10('.', split = 'unlabeled',
                                   transform = transforms.ToTensor(), download = True)
        train_unlabeled_ds = Subset(train_unlabeled_ds, torch.arange(500))
        train_unlabeled_loader = DataLoader(train_unlabeled_ds, batch_size=64)
        return [train_labeled_loader, train_unlabeled_loader]

    def val_dataloader(self):
        # makes the validation dataloader
        val_ds = STL10('.', split = 'test', transform = transforms.ToTensor(), download = True)
        val_ds = Subset(val_ds, torch.arange(500))
        val_loader = DataLoader(val_ds, batch_size=64)
        return val_loader

    def forward(self, x):
        if self.current_model == 'teacher':
            return self.teacher_model.forward(x)
        else:
            return self.student_model.forward(x)

    def training_step(self, batch, batch_nb):
        labeled_batch, unlabeled_batch = batch[0], batch[1]
        if self.current_model == 'teacher':
            labeled_x, labeled_y = labeled_batch
            y_hat = self.teacher_model(labeled_x)
            loss = F.cross_entropy(y_hat, labeled_y)
            # calculating the cross entropy loss on the result
            self.logger.experiment.track(loss.item(), name='train_loss',
                                         stage = self.current_model)
            # logging the loss with "train_" prefix
            return loss
        else:
            labeled_x, labeled_y = labeled_batch
            y_hat = self.student_model(labeled_x)
            labeled_loss = F.cross_entropy(y_hat, labeled_y)
            # calculating the cross entropy loss on the labeled data
            unlabeled_x, _ = unlabeled_batch
            unlabeled_y = self.teacher_model(unlabeled_x)
            unlabeled_y_hat = self.student_model(unlabeled_x)
            unlabeled_loss = F.cross_entropy(unlabeled_y_hat, torch.argmax(unlabeled_y, dim=1))
            loss = labeled_loss + 0.5 * unlabeled_loss
            self.aim_logger.experiment.track(loss.item(), name='train_loss',
                                             stage = self.current_model)
            # logging the loss with "train_" prefix
            return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        if self.current_model == 'teacher':
            y_hat = self.teacher_model(x)
        else:
            y_hat = self.student_model(x)
        loss = F.cross_entropy(y_hat, y)
        # calculating the cross entropy loss on the result
        self.total_classified += y.shape[0]
        self.correctly_classified += (y_hat.argmax(1) == y).sum().item()
        # Calculating total and correctly classified images to determine the accuracy later
        self.aim_logger.experiment.track(loss.item(), name='val_loss',
                                            stage = self.current_model)
        # logging the loss and the accuracy with "val_" prefix
        return loss

    def validation_epoch_end(self, results):
        accuracy = self.correctly_classified / self.total_classified
        self.aim_logger.experiment.track(accuracy, name='accuracy',
                                         stage = self.current_model)
        # logging accuracy
        self.total_classified = 0
        self.correctly_classified = 0
        return accuracy

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def fit_model(self):
        self.current_model = 'teacher'
        self.teacher_trainer.fit(self)

        self.current_model = 'student'
        self.total_classified = 0
        self.correctly_classified = 0
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.student_trainer.fit(self)

if __name__ == "__main__":

    model = ImageClassifierModel()
    model.fit_model()
