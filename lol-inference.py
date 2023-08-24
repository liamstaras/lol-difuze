import torch
import numpy as np
import os
import cmocean
import argparse
from datetime import datetime
from socket import gethostname

import difuze

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('path', type=str, metavar='model path')
parser.add_argument('--batch-size', type=int, default=4)

args = parser.parse_args()

## IMPORTANT HYPERPARAMETERS ##
PATH = args.path
BATCH_SIZE = args.batch_size

# determine whether we will use the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# establish base path
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
base_path = os.path.join('outputs', timestamp+'_'+gethostname())

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

# load the state dict
checkpoint_state_dict = torch.load(PATH)

# make noise schedules
inference_noise_schedule = difuze.support.NoiseSchedule(300, 1e-4, 0.09, np.linspace)

# make datasets
inference_dataset = difuze.data.NpyDataset(
    'data/full-log-clean.npy',
    gt_index=1,
    cond_index=0,
    start_index=-10,
    stop_index=0
)

# make dataloader
inference_dataloader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    use_tensorboard=False, # not required for inference
    visual_function=visual_function,
    save_functions=save_functions,
    base_directory_override='output'
)

# wrap everything up in the inference framework
framework = difuze.inference.InferenceFramework(
    device=device,
    model=model,
    checkpoint_state_dict=checkpoint_state_dict,
    inference_dataloader=inference_dataloader,
    inference_noise_schedule=inference_noise_schedule,
    data_logger=data_logger
)

framework.infer_all_data()