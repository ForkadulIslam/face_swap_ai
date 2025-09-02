import argparse
import datetime
import logging
import math
import random
import time
import torch
import platform
from os import path as osp
import warnings

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (
    MessageLogger, check_resume, get_env_info, get_root_logger, init_tb_logger,
    init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed
)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

# ----------- DEVICE SELECTION ----------
def select_device(prefer_coreml=True):
    if torch.backends.mps.is_available() and prefer_coreml and platform.system() == "Darwin":
        print("BasicSR: Using CoreML backend (MPS).")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("BasicSR: Using CUDA backend.")
        return torch.device("cuda")
    else:
        print("BasicSR: Using CPU backend.")
        return torch.device("cpu")

DEVICE = select_device(prefer_coreml=True)

# ignore UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
warnings.filterwarnings("ignore", category=UserWarning)

def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train)

    # distributed settings
    if args.launcher == 'none' or DEVICE.type != 'cuda':
        opt['dist'] = False
        print('Distributed training disabled.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project') is not None):
        assert opt['logger'].get('use_tb_logger') is True
        init_wandb_logger(opt)

    tb_logger = None
    if opt['logger'].get('use_tb_logger'):
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger

def create_train_val_dataloader(opt, logger):
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=train_sampler, seed=opt['manual_seed'])
            num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(f'Training stats:\n\tTrain images: {len(train_set)}\n\tEnlarge ratio: {dataset_enlarge_ratio}\n\tBatch/GPU: {dataset_opt["batch_size_per_gpu"]}\n\tGPUs: {opt["world_size"]}\n\tIters/epoch: {num_iter_per_epoch}\n\tTotal epochs: {total_epochs}, Iters: {total_iters}')

        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Validation items in {dataset_opt["name"]}: {len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def train_pipeline(root_path):
    opt = parse_options(root_path, is_train=True)

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if opt['path'].get('resume_state'):
        resume_state = torch.load(opt['path']['resume_state'], map_location=DEVICE)
    else:
        resume_state = None

    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    logger, tb_logger = init_loggers(opt)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = create_train_val_dataloader(opt, logger)

    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = build_model(opt).to(DEVICE)
        model.resume_training(resume_state)
        logger.info(f"Resuming from epoch {resume_state['epoch']}, iter {resume_state['iter']}")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = build_model(opt).to(DEVICE)
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu' or DEVICE.type in ['cpu', 'mps']:
        if prefetch_mode == 'cuda' and DEVICE.type == 'mps':
            logger.warning("CUDA prefetch requested but MPS (CoreML) is in use. Falling back to CPU prefetch.")
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        if DEVICE.type != 'cuda':
            logger.warning("CUDA prefetch requested but CUDA unavailable. Using CPU prefetch.")
            prefetcher = CPUPrefetcher(train_loader)
        else:
            if opt['datasets']['train'].get('pin_memory') is not True:
                raise ValueError('Set pin_memory=True for CUDAPrefetcher.')
            prefetcher = CUDAPrefetcher(train_loader, opt)
            logger.info(f'Using CUDA prefetcher')
    else:
        raise ValueError(f"Invalid prefetch_mode: {prefetch_mode}. Supported: 'cpu', 'cuda', None")

    logger.info(f'Start training at epoch {start_epoch}, iter {current_iter + 1}')
    start_time = time.time()
    data_time, iter_time = time.time(), time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time
            current_iter += 1
            if current_iter > total_iters:
                break

            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving model and training state.')
                model.save(epoch, current_iter)

            if opt.get('val') and opt['datasets'].get('val') and (current_iter % opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'Training complete. Time: {consumed_time}')
    logger.info('Saving latest model.')
    model.save(epoch=-1, current_iter=-1)

    if opt.get('val') and opt['datasets'].get('val'):
        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)