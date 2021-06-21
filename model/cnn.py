import argparse
import pytorch_lightning as pl
from utils import recursive_find_python_class, random_seed_
from model.densenet import *
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch import nn, optim

class CNNSurvival(pl.LightningModule):

    def __init__(self, model='densenet121', out_size=2, norm='in', lr=1e-4):
        super().__init__()
        self.lr = lr
        model_fn = recursive_find_python_class(['model'], model, current_module = 'model')
        self.model = model_fn(num_classes=out_size, norm=norm)

    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1,1,1))
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)
        return out

    def training_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = batch['label']

        out = self.model(x)
        loss = F.cross_entropy(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = batch['label']

        out = self.model(x)
        loss = F.cross_entropy(out, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = batch['label']

        out = self.model(x)
        loss = F.cross_entropy(out, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        return parser