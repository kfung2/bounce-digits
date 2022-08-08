import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *

# Configs
batch_size = 16
learning_rate = 1e-3
epochs = 100

num_ctx_frames = 5
num_tgt_frames = 5

hid_s=64
hid_t=256
N_s=4
N_t=8
kernel_sizes=[11,11,11,11]
groups=4

channels = 3
height = 128
width = 128
input_shape = (channels, num_ctx_frames, height, width)

model = SimVP(input_shape=input_shape, 
              hid_s=hid_s, hid_t=hid_t, 
              N_s=N_s, N_t=N_t,
              kernel_sizes=kernel_sizes, 
              groups=groups,
              learning_rate=learning_rate)

moving_mnist = TwoColourMovingMNISTDataModule(batch_size, num_ctx_frames, num_tgt_frames)

logger = TensorBoardLogger('./logs', 'SimVP_RGB')

trainer = pl.Trainer(gpus=4, 
                     strategy=DDPStrategy(find_unused_parameters=False),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, moving_mnist)