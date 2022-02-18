import os
import math
from re import T
import time
from numpy.lib import utils
import torch
import argparse
import json
import pdb
import random
import shutil
from torch import device, dtype, float32, tensor
import torch.nn as nn
from sklearn.manifold import TSNE
import numpy as np
from model.ae.lstm_ae import LSTM_Embedding
from model.ae.GMVAE import LatentGaussianMixture
from model.ae.GMVAE import GMMModel
from model.ae.lstm_enc_dec import LSTMEDModule
from dataset.dataset_train import DatasetTrain
from dataset.dataset_train_aggre import DatasetTrainAggre
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from trainer import Trainer
from utils import AverageMeter
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from model.embedding.gener_embedding import Gener_embedding
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


with open("./config.json") as f:
    config = json.load(f)

def collate_fn(data):
    cuda_condition = torch.cuda.is_available() 
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    # print("cillo")
    # print([len(s[0]) for s in data])
    # print([data_seq[1] for data_seq in data])
    filename = []
    
    data.sort(key=lambda x: len(x[0]), reverse=True)
    seq_len = [len(s[0]) for s in data]
    label = [data_seq[1] for data_seq in data]
    filename = [data_seq[2] for data_seq in data]
    data_info = [data_seq[0] for data_seq in data]
    #filename = [data_seq[2] for data_seq in data]
    data_out_pad = pad_sequence(data_info, batch_first=True, padding_value=0)
    return [data_out_pad,label,seq_len,filename]
    
    

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default='save/', help="ex)output/bert.model")
    parser.add_argument("-hs", "--h_dims", type=int, default=64, help="hidden size of lstm model")
    parser.add_argument("-cn", "--cluster_num", type=int, default=5, help="cluster_num of mixgauss")
    parser.add_argument("-o", "--output_dim", type=int, default=64, help="output size of lstm model")
    parser.add_argument("-i", "--input_dim", type=int, default=2, help="input size of lstm model")
    parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers")
    parser.add_argument("-em", "--embed_size", type=int, default=64, help="size of embedding")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="number of epochs") 
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1], help="CUDA device ids")
    parser.add_argument("--lr", type=float, default=0.008, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--train_mode", type=int, default=0, help="0)train and test, 1)pretrain task1, 2)pretrain task2")
    parser.add_argument("--load_file", type=str, default=None)
    parser.add_argument("--grid", type=bool, default=True ,help="location to grid")
    parser.add_argument("-d", "--data_size", type=int, default=10000, help="number of epochs")
    parser.add_argument("-sn", "--segment_num", type=int, default=10, help="number of epochs")
    args = parser.parse_args()
    print("args_train",args)
    
    if args.train_mode == 0 or args.train_mode == 1:
        print("Loading Train Dataset", config['train_dataset'])
        train_dataset = DatasetTrain(config, config['train_dataset'], train=True, grid=True)
        print("Loading Test Dataset", config['temp_dataset'])
        test_dataset = DatasetTrain(config, config['temp_dataset'], train=False, grid=True)
        print("Loading Val Dataset", config['val_nor_dataset'])
        val_nor_dataset = DatasetTrain(config, config['temp_dataset'], train=False, grid=True)
        print("Loading Val Dataset", config['val_ab_dataset'])
        val_ab_dataset = DatasetTrain(config, config['temp_dataset'], train=False, grid=True)
   
    
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn, num_workers=args.num_workers)
    #print(train_data_loader)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,collate_fn=collate_fn, num_workers=args.num_workers) \
        if test_dataset is not None else None
    val_nor_loader = DataLoader(val_nor_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn, num_workers=args.num_workers)
    val_ab_loader = DataLoader(val_ab_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn, num_workers=args.num_workers)

    print("Building model")
    model = GMMModel(args, config,args.input_dim, args.output_dim, args.h_dims)

    ref_grade = []
    x_grad = AverageMeter()
    x_grad.avg = torch.zeros(OrderedDict(model.decoder.layer.named_parameters())["weight_ih_l0"].shape).cuda()
    h_grad = AverageMeter()
    h_grad.avg = torch.zeros(OrderedDict(model.decoder.layer.named_parameters())["weight_hh_l0"].shape).cuda()
    ref_grade.append(x_grad)
    ref_grade.append(h_grad)

    print("Creating Trainer")
    if args.train_mode == 0 or args.train_mode == 1:
        trainer =Trainer(model,hidden_size = args.h_dims,output_size=args.output_dim, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                            lr=args.lr, weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, batch_size = args.batch_size,
                            train_mode = args.train_mode, load_file = args.load_file, output_path = args.output_path, config = config)
    
    
        os.system("rm " + config['result_file'] + "_" + str(args.train_mode) + '_test.txt')
        os.system("rm " + config['result_file'] + "_" + str(args.train_mode) +'_train.txt')
        os.system("touch " + config["result_file"] + "_" + str(args.train_mode) + '_test.txt')
        os.system("touch " + config["result_file"] + "_" + str(args.train_mode) + '_train.txt')
       
        # init_data = pack_padded_sequence(init, init_len, batch_first=True)
        writer = SummaryWriter(log_dir="runs")
        print("Training Start")
        for epoch in range(args.epochs):
            with torch.backends.cudnn.flags(enabled=False):
                trainer.train(writer,epoch,ref_grade)
                trainer.save(epoch, args.output_path)
        writer.close()
        torch.save(ref_grade,".ref_grad.pth")

if __name__ == '__main__':
    train()