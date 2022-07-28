import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.threed_conv_classes import *

class ThreeDConv(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()

        self.mod = ThreeDConvWideTwoDeepTwo()
        self.mse = nn.MSELoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        out = self.mod(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        return {"optimizer": optimizer, 
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss"
                }

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.mse(outputs, y)
        self.log('train_loss', loss, 
                 on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.mse(outputs, y)

        self.log_dict({"val_loss": loss}, on_step=False, on_epoch=True, prog_bar=False)         
        return loss
        
        
