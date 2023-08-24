import torch
import numpy as np
import os
import cmocean
import argparse
from datetime import datetime
from socket import gethostname

import difuze

from metrics.cosmology import PixelCounts, PeakCounts, PowerSpectrum
from metrics.metrics import RelativeDifference

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--initial-learning-rate', type=float, default=1e-4)
parser.add_argument('--gamma-decay', type=float, default=0.9)
parser.add_argument('--loss-function', type=str, default='L1Loss')

args = parser.parse_args()

## IMPORTANT HYPERPARAMETERS ##
BATCH_SIZE = args.batch_size
INITIAL_LEARNING_RATE = args.initial_learning_rate
LEARNING_RATE_GAMMA_DECAY = args.gamma_decay
LOSS_FUNCTION = args.loss_function

# determine whether we will use the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# establish base path
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
base_path = os.path.join('runs', timestamp+'_'+gethostname())

# create the model
model = difuze.models.Palette(
    image_size=128,
    in_channel=2,
    out_channel=1,
    inner_channel=64,
    channel_mults=(1,2,4,8),
    attn_res=(16,),
    num_head_channels=32,
    res_blocks=2,
    dropout=0.2
)

# create the loss function, optimizer and scheduler
if LOSS_FUNCTION == 'L1Loss':
    training_loss_function = torch.nn.L1Loss()
elif LOSS_FUNCTION == 'MSELoss':
    training_loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    params=list(filter(lambda p: p.requires_grad, model.parameters())), 
    lr=INITIAL_LEARNING_RATE, 
    weight_decay=0
)

loss_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer,
    gamma=LEARNING_RATE_GAMMA_DECAY
)

metric_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=0.5,
    patience=3,
    threshold=0.05
)

# make noise schedules
training_noise_schedule = difuze.support.NoiseSchedule(2000, 1e-6, 0.01, np.linspace)
inference_noise_schedule = difuze.support.NoiseSchedule(300, 1e-4, 0.09, np.linspace)

# make datasets
training_dataset = difuze.data.NpyDataset(
    'data/full-log-clean.npy',
    gt_index=1,
    cond_index=0,
    stop_index=0.899
)
evaulation_dataset = difuze.data.NpyDataset(
    'data/full-log-clean.npy',
    gt_index=1,
    cond_index=0,
    start_index=0.899,
    stop_index=0.90
)

# prepare evaluation metrics

statistics = [
    PixelCounts(), PeakCounts(), PowerSpectrum()
]
evaluation_metrics = [
    difuze.metrics.NumpyMetric(
        RelativeDifference(
            statistic
        )
    ) for statistic in statistics]

# make dataloaders
training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
evaluation_dataloader = DataLoader(evaulation_dataset, batch_size=BATCH_SIZE, shuffle=False)

# make function for generating images
def tensor_to_image_cmocean(tensor):
    cmin = -2
    cmax = 2
    normalized = (torch.clamp(tensor, cmin, cmax)-cmin)/(cmax-cmin) # now in range (0,1)
    image = cmocean.cm.deep_r(normalized.cpu().numpy())
    # roll axes and cut alpha channel
    return image.transpose((2,0,1))[:3]

# specify functions to save outputs
visual_function = tensor_to_image_cmocean
save_functions = [
    difuze.data.NpySaver(),
    difuze.data.TifSaver(tensor_to_image_cmocean)
]

# initialize the data logger
data_logger = difuze.log.DataLogger(
    use_tensorboard=True,
    visual_function=visual_function,
    save_functions=save_functions
)

# wrap everything up in the training framework
framework = difuze.training.TrainingFramework(
    device=device,
    model=model,
    optimizer=optimizer,
    loss_scheduler=loss_scheduler,
    training_dataloader=training_dataloader,
    training_noise_schedule=training_noise_schedule,
    training_loss_function=training_loss_function,
    evaluation_dataloader=evaluation_dataloader,
    inference_noise_schedule=inference_noise_schedule,
    evaluation_metrics=evaluation_metrics,
    data_logger=data_logger,

    metric_scheduler=metric_scheduler
)

# run training
framework.main_training_loop()