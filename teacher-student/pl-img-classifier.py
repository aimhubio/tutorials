import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.models import resnet18
from torchvision import transforms
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger

class ImageClassifierModel(pl.LightningModule):

    def __init__(self):
        super(ImageClassifierModel, self).__init__()
        self.total_classified = 0
        self.is_on_teacher = True
        self.correctly_classified = 0
        self.teacher_model = resnet18(pretrained = True, progress = True)
        self.student_model = resnet18(pretrained = True, progress = True)
        self.teacher_model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        self.student_model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        # changing the last layer from 1000 out_features to 10 because the model
        # is pretrained on ImageNet which has 1000 classes but STL10 has 10
        self.make_trainers()
        # self.model.to('cuda') # moving the model to cuda

    def make_trainers(self):
        aim_teacher_logger = AimLogger(
            experiment='resnet18_classification',
            train_metric_prefix='train_',
            val_metric_prefix='val_',
        )
        self.teacher_trainer = pl.Trainer(
            logger=aim_teacher_logger,
            gpus=0,
            max_epochs=1,
            progress_bar_refresh_rate=1,
            log_every_n_steps=10,
            check_val_every_n_epoch=1)

        aim_student_logger = AimLogger(
            experiment='resnet18_classification',
            train_metric_prefix='train_',
            val_metric_prefix='val_',
        )
        self.student_trainer = pl.Trainer(
            logger=aim_student_logger,
            gpus=0,
            max_epochs=1,
            progress_bar_refresh_rate=1,
            log_every_n_steps=10,
            check_val_every_n_epoch=1)

    def set_is_on_teacher(self, val):
        self.is_on_teacher = val

    def train_dataloader(self):
        # makes the training dataloaders
        train_labeled_ds = STL10('.', split = 'train', transform = transforms.ToTensor(), download = True)
        train_labeled_loader = DataLoader(train_labeled_ds, batch_size=8)
        train_unlabeled_ds = STL10('.', split = 'unlabeled', transform = transforms.ToTensor(), download = True)
        train_unlabeled_loader = DataLoader(train_unlabeled_ds, batch_size=8)
        return [train_labeled_loader, train_unlabeled_loader]

    def val_dataloader(self):
        # makes the validation dataloader
        val_ds = STL10('.', split = 'test', transform = transforms.ToTensor(), download = True)
        val_loader = DataLoader(val_ds, batch_size=8)
        return val_loader

    def forward(self, x):
        if self.is_on_teacher:
            return self.teacher_model.forward(x)
        else:
            return self.student_model.forward(x)

    def training_step(self, batch, batch_nb):
        labeled_batch, unlbeled_batch = batch[0], batch[1]
        if self.is_on_teacher:
            labeled_x, labeled_y = labeled_batch
            y_hat = self.teacher_model(labeled_x)
            loss = F.cross_entropy(y_hat, labeled_y)
            # calculating the cross entropy loss on the result
            self.log('train_loss', loss)
            # logging the loss with "train_" prefix
            return loss
        else:
            labeled_x, labeled_y = labeled_batch
            y_hat = self.student_model(labeled_x)
            labeled_loss = F.cross_entropy(y_hat, labeled_y)
            # calculating the cross entropy loss on the labeled data
            unlabeled_x = unlbeled_batch
            unlabeled_y = self.teacher_model(unlabeled_x)
            unlabeled_y_hat = self.student_model(unlabeled_x)
            unlabeled_loss = F.cross_entropy(unlabeled_y_hat, unlabeled_y)
            loss = labeled_loss + unlabeled_loss
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
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def fit_model(self):
        self.is_on_teacher = True
        self.teacher_trainer.fit(self)

        self.is_on_teacher = False
        self.total_classified = 0
        self.correctly_classified = 0
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.student_trainer.fit(self)

if __name__ == "__main__":

    model = ImageClassifierModel()
    model.fit_model()
