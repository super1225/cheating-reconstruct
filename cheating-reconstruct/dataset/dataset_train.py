import os
import torch
import json
import csv
import pdb
from torch.utils.data import Dataset
import numpy as np
import random
import math
import copy

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .utils import append_angle_displacement

class DatasetTrain(Dataset):  

    def __init__(self, config, file_directory, len_threshold=20, train=False, grid=True):
        self.config = config
        self.grid = grid
        self.files = os.listdir(file_directory)
        self.file_directory = file_directory
        self.max_length = config['max_len']
        self.len_threshold = len_threshold

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):  

        while (True):  
            with open(self.file_directory + "/" + self.files[item]) as f:
                if(int(self.files[item][-6]) == 0):
                    label = 0
                else:
                    label = 1
                line = json.load(f)
                filename = self.files[item]
            if len(line[1]) > self.len_threshold:
                break
            else:
                item = (item + 1) % self.__len__()
        line_temp = copy.deepcopy(line)
        
        
        
        line_o = append_angle_displacement(line, disturb=True, disturb_angle=5, disturb_disp=500)
        if self.grid:
            for i in range(0,len(line_temp[1])):
                if line_temp[0][1] == line_temp[0][3]:
                        yuzhi_y = 1
                else:
                    yuzhi_y = line_temp[0][1] - line_temp[0][3]
                
                y = (line_temp[1][i][1]-line_temp[0][3])/(yuzhi_y)

                if line_temp[0][0] == line_temp[0][2]:
                    yuzhi_x = 1
                else:
                    yuzhi_x = line_temp[0][0]-line_temp[0][2]
              
                x = (line_temp[1][i][0]-line_temp[0][2])/(yuzhi_x)
                line_o[i][0] = x
                line_o[i][1] = y
                if i > 0:
                    line_o[i][2] = abs(line_temp[1][i][2]-line_temp[1][i-1][2])
                    if line_o[i][2]>60000:
                        line_o[i][2] = 60000
                else:
                    line_o[i][2] = 0
        
        line_o = torch.tensor([position[0:2] for position in line_o], dtype=torch.float64)
        return [line_o,label,filename]


    def to_grids(self, max_x, max_y, point):
        """
        :param max_width: the maximum value of x-axis coordinate
        :param max_height: the maximum value of y-axis coordinate
        :param r_x: the width of each grid
        :param r_y: the height of each grid
        """
        return self.config['y_grid_nums']*int(self.config['x_grid_nums']*point[0]/max_x)+int(self.config['y_grid_nums']*point[1]/max_y)
