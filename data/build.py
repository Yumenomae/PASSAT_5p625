import os
import torch
import numpy as np
import torch.distributed as dist

from .data_folder import DatasetFolder

def build_loader(config, local_rank):
    dataset_train = build_dataSet(config=config, type='train')
    print(f"local rank {local_rank} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val = build_dataSet(config=config, type='valid')
    print(f"local rank {local_rank} / global rank {dist.get_rank()} successfully build val dataset")
    dataset_test = build_dataSet(config=config, type='test')
    print(f"local rank {local_rank} / global rank {dist.get_rank()} successfully build test dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test

def build_dataSet(config, type):
    root = './Lists/dataList_1979_2018_' + type 
    dataSet = DatasetFolder(dataRoot=root, dataMean='./Storages/DataStat/dataMean', 
                            dataStd='./Storages/DataStat/dataStd', windowSize=config.DATA.WINDOWSIZE)
    return dataSet

