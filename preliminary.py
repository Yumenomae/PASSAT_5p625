import os
from data import make_dataList, make_constants, make_dataStat, seperate_dataList

if __name__ == '__main__':

    # path = 'The path of the ERA5 dataset'
    path = './data/ERA5'

    if os.path.exists('Outputs'):
        print('saving models into Outputs')
    else: os.mkdir('Outputs')

    if os.path.exists('Storages'):
        print('get data and embeds from Storages')
    else: os.mkdir('Storages')

    if os.path.exists('Lists'):
        print('load data from Lists')
    else: os.mkdir('Lists')

    startYear, endYear = 1979, 2018
    periods = [i for i in range(int(startYear), int(endYear)+1)]
    make_dataList(path, periods) # Generate the data storage from given year
    seperate_dataList()
    make_constants(path)    # Generate the embedding storage from 2000 year
    make_dataStat(path, '2006')   # Generate the mean and standard error from 2006 year