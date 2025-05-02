import io
import os
import time
import torch
import numpy as np
import torch.distributed as dist
import torch.utils.data as data

class DatasetFolder(data.Dataset):

    def __init__(self, dataRoot, dataMean, dataStd, windowSize):
        # image folder mode
        self.root = dataRoot
        self.dataList = torch.load(dataRoot)
        self.dataMean = torch.load(dataMean)[:, None, None] # (5,1,1)
        self.dataStd = torch.load(dataStd)[:, None, None]   # (5,1,1)
        self.windowSize = windowSize

    def __getitem__(self, index):

        dataPath = self.dataList[index] # e.g., ./data_storage/yyyy_idx_data
        targetPath = self.dataList[index + 1 : index + self.windowSize]

        data = torch.load(dataPath)  # (5,32,64)
        data = (data - self.dataMean) / self.dataStd
        dataInfo = [int(dataPath.split('_')[1])]
        target = []
        for path in targetPath:
            dataInfo.append(int(path.split('_')[1]))
            subTarget = torch.load(path)
            subTarget = (subTarget - self.dataMean) / self.dataStd
            target.append(subTarget)

        return data.float(), torch.tensor(dataInfo), torch.stack(target).float()

    def __len__(self):
        return len(self.dataList) - self.windowSize + 1 # to ensure every sample has target.

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
