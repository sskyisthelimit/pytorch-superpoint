"""many loaders
# loader for model, dataset, testing dataset
"""

import os
import logging
from pathlib import Path

import requests
import numpy as np
import torch
import torch.optim
import torch.utils.data


from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
# from settings import EXPER_PATH

# from utils.loader import get_save_path
def get_save_path(output_dir):
    """
    This func
    :param output_dir:
    :return:
    """
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.

   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

   """
   base_seed = torch.IntTensor(1).random_().item()
   # print(worker_id, base_seed)
   np.random.seed(base_seed + worker_id)


def dataLoader(config, dataset='syn', warp_input=False, train=True, val=True):
    import torchvision.transforms as transforms
    training_params = config.get('training', {})
    workers_train = training_params.get('workers_train', 1) # 16
    workers_val   = training_params.get('workers_val', 1) # 16
        
    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}")
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
    # if dataset == 'syn':
    #     from datasets.SyntheticDataset_gaussian import SyntheticDataset as Dataset
    # else:
    Dataset = get_module('datasets', dataset)
    print(f"dataset: {dataset}")

    train_set = Dataset(
        transform=data_transforms['train'],
        task = 'train',
        **config['data'],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['model']['batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_train,
        worker_init_fn=worker_init_fn
    )
    val_set = Dataset(
        transform=data_transforms['train'],
        task = 'val',
        **config['data'],
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config['model']['eval_batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_val,
        worker_init_fn=worker_init_fn
    )
    # val_set, val_loader = None, None
    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}

def dataLoader_test(config, dataset='syn', warp_input=False, export_task='train'):
    import torchvision.transforms as transforms
    training_params = config.get('training', {})
    workers_test = training_params.get('workers_test', 1) # 16
    logging.info(f"workers_test: {workers_test}")

    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }
    test_loader = None
    if dataset == 'syn':
        from datasets.SyntheticDataset import SyntheticDataset
        test_set = SyntheticDataset(
            transform=data_transforms['test'],
            train=False,
            warp_input=warp_input,
            getPts=True,
            seed=1,
            **config['data'],
        )
    elif dataset == 'hpatches':
        from datasets.patches_dataset import PatchesDataset
        if config['data']['preprocessing']['resize']:
            size = config['data']['preprocessing']['resize']
        test_set = PatchesDataset(
            transform=data_transforms['test'],
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn
        )
    # elif dataset == 'Coco' or 'Kitti' or 'Tum':
    else:
        # from datasets.Kitti import Kitti
        logging.info(f"load dataset from : {dataset}")
        Dataset = get_module('datasets', dataset)
        test_set = Dataset(
            export=True,
            task=export_task,
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn

        )
    return {'test_set': test_set, 'test_loader': test_loader}

def get_module(path, name):
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)

def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)

def modelLoader(model='SuperPointNet', **options):
    # create model
    logging.info("=> creating model: %s", model)
    net = get_model(model)
    net = net(**options)
    return net


# mode: 'full' means the formats include the optimizer and epoch
# full_path: if not full path, we need to go through another helper function
def xavier_init_weights(module):
    """Initialize weights of a module using Xavier initialization."""
    if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)


def pretrainedLoader(net, optimizer, epoch, path, mode='full', full_path=False):
    
    url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
    checkpoint_path = "superpoint_v1.pth"
    response = requests.get(url)
    with open(checkpoint_path, "wb") as f:
        f.write(response.content)
    # Load checkpoint
    
    if full_path:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = load_checkpoint(checkpoint_path)
    
    # Handle missing keys with Xavier initialization
    model_state = net.state_dict()
    if mode == 'full':
        checkpoint_state = checkpoint['model_state_dict']
    else:
        checkpoint_state = checkpoint
    
    missing_keys, unexpected_keys = net.load_state_dict(checkpoint_state, strict=False)
    
    if missing_keys:
        logging.info(f"Missing keys in checkpoint: {missing_keys}")
        for key in missing_keys:
            if key in model_state:
                module_name = key.split('.')[0]
                module = getattr(net, module_name, None)
                if module is not None:
                    module.apply(xavier_init_weights)
    
    if mode == 'full':
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['n_iter']
    else:
        epoch = 0

    return net, optimizer, epoch

if __name__ == '__main__':
    net = modelLoader(model='SuperPointNet')

