import logging
from re import A

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from .base_model import Encoder,Decoder

class LSTMEDModule(nn.Module):
    def __init__(self, n_features: int=2, hidden_size: int=64,
                 n_layers: tuple=(1,1), use_bias: tuple=(True,True), dropout: tuple=(0,0),):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool = False):
        batch_size = ts_batch.shape[0]
        enc_hidden = self._init_hidden(batch_size)  
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  
        dec_hidden = enc_hidden
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output



class LatentGaussianMixture:
    def __init__(self, args):
        self.args = args
        self.mu_c = torch.Tensor(args.cluster_num, args.h_dims)
        torch.nn.init.uniform_(self.mu_c,a=0,b=1)
        self.log_sigma_sq_c = torch.Tensor(args.cluster_num, args.h_dims)
        torch.nn.init.constant_(self.log_sigma_sq_c,0.0)
        
        self.fc_mu_z = torch.nn.Linear(args.h_dims,args.h_dims)
        initial_fc_mu_z_Wight = torch.Tensor(args.h_dims, args.h_dims).cuda()
        torch.nn.init.normal_(initial_fc_mu_z_Wight,mean=0,std=0.02)
        initial__fc_mu_z_Bias = torch.Tensor(args.h_dims).cuda()
        torch.nn.init.constant_(initial__fc_mu_z_Bias,0.0)
        self.fc_mu_z.weight = torch.nn.Parameter(initial_fc_mu_z_Wight)
        self.fc_mu_z.bias = torch.nn.Parameter(initial__fc_mu_z_Bias)

        self.fc_sigma_z = torch.nn.Linear(args.h_dims,args.h_dims)
        initial_fc_sigma_z_Wight = torch.Tensor(args.h_dims, args.h_dims).cuda()
        torch.nn.init.normal_(initial_fc_sigma_z_Wight,mean=0,std=0.02)
        initial_fc_sigma_z_Bias = torch.Tensor( args.h_dims).cuda()
        torch.nn.init.constant_(initial_fc_sigma_z_Bias,0.0)
        self.fc_sigma_z.weight = torch.nn.Parameter(initial_fc_sigma_z_Wight) 
        self.fc_sigma_z.bias = torch.nn.Parameter(initial_fc_sigma_z_Bias) 

    def post_sample(self, embeded_state, return_loss=False):
        args = self.args
        if(len(embeded_state.shape)==1):
            embeded_state = embeded_state.unsqueeze(0)
        if embeded_state.shape == 3:
            #print(embeded_state.shape[0])
            embeded_state = embeded_state[embeded_state.shape[0]-1]
        mu_z = self.fc_mu_z(embeded_state)
        log_sigma_sq_z = self.fc_sigma_z(embeded_state)
        eps_z = torch.Tensor(log_sigma_sq_z.shape).cuda()
        eps_z = torch.nn.init.normal_(eps_z, mean=0.0, std=1.0)
        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z
        stack_z = torch.stack([z] * args.cluster_num, axis=1).cuda()
        
        stack_mu_c = torch.stack([self.mu_c] * z.shape[0], axis=0).cuda()
        stack_mu_z = torch.stack([mu_z] * args.cluster_num, axis=1).cuda()
        stack_log_sigma_sq_c = torch.stack([self.log_sigma_sq_c] * z.shape[0], axis=0).cuda()
        stack_log_sigma_sq_z = torch.stack([log_sigma_sq_z] * args.cluster_num, axis=1).cuda()
       
        pi_post_logits = - torch.sum(torch.square(stack_z - stack_mu_c) / torch.exp(stack_log_sigma_sq_c), dim=-1)
        tosoftmax = nn.Softmax(dim=1)
        pi_post = tosoftmax(pi_post_logits) + 1e-10

        if not return_loss:
            return z
        else:
            batch_gaussian_loss = 0.5 * torch.sum(
                    pi_post * torch.mean(stack_log_sigma_sq_c
                        + torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c)
                        + torch.square(stack_mu_z - stack_mu_c) / torch.exp(stack_log_sigma_sq_c), dim=-1)
                    , dim=-1) - 0.5 * torch.mean(1 + log_sigma_sq_z, dim=-1)

            batch_uniform_loss = torch.abs(torch.mean(torch.mean(pi_post, dim=0) * torch.log(torch.mean(pi_post, dim=0))))
            return z, [batch_gaussian_loss, batch_uniform_loss],mu_z,log_sigma_sq_z

    def prior_sample(self):
        pass



class GMMModel(nn.Module):
    def __init__(self,args, config,input_dim, output_dim, h_dims=[], h_activ=nn.Sigmoid(), out_activ=nn.Tanh()):
        super().__init__()
        self.args = args
        self.encoder = Encoder(input_dim, output_dim, h_dims, h_activ,
                               out_activ)
        self.decoder = Decoder(input_dim, output_dim,h_dims, h_activ,out_activ)
        self.classsifier = nn.Linear(h_dims,2)

        self.latent_space = LatentGaussianMixture(args)
        self.out_w = torch.Tensor(args.batch_size, h_dims)
        torch.nn.init.normal_(self.out_w,mean=0,std=0.02)
        self.out_b = torch.Tensor(args.batch_size)
        torch.nn.init.constant_(self.out_b,0.0)
        self.loss_mse = torch.nn.MSELoss(reduction='mean').cuda()
        
    def forward(self, inputs,sequence_len):
        args = self.args
        encoder_final_state,out = self.encoder(inputs)
        input_padding,seq_lengths = pad_packed_sequence(inputs)

      
        z, latent_losses,mu_z,log_sigma_sq_z = self.latent_space.post_sample(encoder_final_state[0].squeeze(), return_loss=True)
        outputs,data_orig,sequence_len = self.decoder(inputs,z,encoder_final_state)
        res_loss, res_pretrain_loss = self.loss(outputs, data_orig,sequence_len,latent_losses)
        abnormal = self.classsifier(z)
        res_loss, res_pretrain_loss = self.loss(outputs, data_orig,sequence_len,latent_losses)
        return res_loss, res_pretrain_loss,abnormal,mu_z,log_sigma_sq_z,z,outputs,data_orig,sequence_len
  

    def loss(self, outputs, data_orig,sequence_len,latent_losses):
        args = self.args
        batch_gaussian_loss, batch_uniform_loss = latent_losses
        mse_loss = []
        mse_loss1 = 0
        for i, traj_len in enumerate(sequence_len):
            mse_loss1 = mse_loss1 + \
                self.loss_mse(outputs[i, :traj_len, :],
                    data_orig[i, :traj_len, :])
        rec_loss = mse_loss1/len(sequence_len)
        gaussian_loss = torch.mean(batch_gaussian_loss)
        uniform_loss = torch.mean(batch_uniform_loss)
        print("*"*100)
        print(100*rec_loss)
        print(100/ self.args.h_dims * gaussian_loss)
        print(uniform_loss)
        if args.cluster_num == 1:
            loss = rec_loss + gaussian_loss
        else:
            loss = 100*rec_loss + 100.0 / self.args.h_dims * gaussian_loss + 1.0 * uniform_loss
          
        pretrain_loss = rec_loss
        return loss, pretrain_loss

