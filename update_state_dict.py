### small utility script to update state_dict to newer format


import torch

IN_PATH = 'runs/2023-08-25_13-15-59_gpu025.baobab/models/model_21_BEST'
OUT_PATH = 'models_to_test/gpu025_Aug25_batch32_MSE_epoch21'

state_dict = torch.load(IN_PATH, map_location=torch.device('cpu'))

epoch_number = state_dict['epoch_number']
model = (state_dict['model_state_dict'])
optimizer = (state_dict['optimizer_state_dict'])
loss_scheduler = (state_dict['loss_scheduler_state_dict'])
recent_rms_metrics = state_dict['recent_rms_metrics']
best_rms_metrics = state_dict['best_rms_metrics']
_initial_learning_rate = state_dict['initial_lr']

out_state_dict = {
    'save_hostname': input('hostname: '),
    'save_timestamp': input('timestamp: '),
    'epoch_number': epoch_number,
    'model_state_dict': model,
    'optimizer_state_dict': optimizer,
    'loss_scheduler_state_dict': loss_scheduler,
    'recent_rms_metrics': recent_rms_metrics,
    'best_rms_metrics': best_rms_metrics,
    'configuration': {
        'batch_size': input('batch size: '),
        'initial_lr': _initial_learning_rate,
        'optimizer': input('optimizer: '),
        'loss_function': input('loss function: ')
    }
}

torch.save(out_state_dict, OUT_PATH)
