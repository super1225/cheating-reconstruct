from shutil import copy
from typing import Sequence
import torch
import torch.nn as nn
from torch.nn.modules.module import T
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import pdb
import numpy as np
import copy
import math
from collections import OrderedDict
from model.ae.lstm_ae import LSTM_Embedding
from model.ae.GMVAE import GMMModel
from model.classifier import Lstmclassifier,LstmclassifierAggragate
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence


class Trainer:

    def __init__(self, model:GMMModel,hidden_size,output_size, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, with_cuda: bool = True, cuda_devices=None, batch_size: int = 30,
                 train_mode: int = 0, load_file: str = None, output_path: str = None, config: dict = None):

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.model = model.cuda()
        self.load_file = load_file

        if load_file != None and load_file[:5] == 'train':  
            self.model.load_state_dict(torch.load(output_path + load_file))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:  
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.loss_crosse = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss_mse = torch.nn.MSELoss(reduction='mean').to(self.device)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.train_mode = train_mode
        self.config = config
        self.batch_size = batch_size

        print("batchsize:",batch_size)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def dataset_renew(self, train_dataloader, test_dataloader):
        self.train_data = train_dataloader
        self.test_data = test_dataloader

    def train(self,writer, epoch,ref_grade):
        self.iteration(writer,epoch,ref_grade, self.train_data)

    def test(self,writer, epoch,ref_grade):
        self.iteration(writer,epoch,ref_grade, self.test_data, train=False)
    
    def val(self,writer, epoch,ref_grade):
        list_loss = []
        list_loss = copy.deepcopy(self.iteration(writer,epoch,ref_grade, self.test_data, train=False))
        return list_loss


    def iteration(self,writer, epoch, ref_grade, data_loader, train=True):
        str_code = "train" if train else "test"
        # Setting the tqdm progress bar 
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        if self.train_mode == 0 or self.train_mode == 1:#0:without anglle and distance  1:with angle and distance  
            mu_list = []
            sigma_list = []
            loss_list = []
            data = [[1,2,3],[6,4,5]]
            loss_list = []
            loss_grad_list = []
            loss_recon_list = []
            for i, data_batch in data_iter:
                data = data_batch[0]
                label = data_batch[1]
                seq_len = data_batch[2]
                filename = data_batch[3]
                data = pack_padded_sequence(data, seq_len, batch_first=True).to(self.device)
                label = torch.tensor(label,dtype=torch.long).to(self.device)
                seq_len_tensor = torch.tensor(seq_len,dtype=torch.float32).to(self.device)

                loss_vae, pretrain_loss,abnormal_detection,mu_z,log_sigma_sq_z,z,outputs,data_orig,sequence_len =  self.model.forward(data.to(torch.float32),seq_len_tensor)     
                loss_vae.requires_grad_() 
                #grad_loss
                model_weights = OrderedDict(self.model.decoder.layer.named_parameters())
                gradients = torch.autograd.grad(loss_vae, model_weights.values(),create_graph=True,retain_graph=True,allow_unused=True)
                gradients_weight = list(gradients[0:2])
                grade_loss = torch.tensor(0)
                first_num = 0
                for k in range(0,len(gradients_weight)):
                    first_num = first_num + sum(ref_grade[k].avg.view(-1,1).squeeze(1).tolist())
                if first_num != 0:
                    for k in range(0,len(gradients_weight)):
                        grade_loss = grade_loss + -1*torch.nn.functional.cosine_similarity(gradients_weight[k].view(-1,1),ref_grade[k].avg.view(-1,1),dim=0).squeeze(0)
                #total loss
                loss = loss_vae + 5*grade_loss    
                loss_list.append(loss.item())
                mu_list.extend(mu_z.tolist())
                sigma_list.extend(log_sigma_sq_z.tolist())  
                if train:
                    self.optim.zero_grad()  
                    loss.backward(retain_graph=True)  
                    x_grad = copy.deepcopy(OrderedDict(self.model.decoder.layer.named_parameters())["weight_ih_l0"].grad)
                    ref_grade[0].update(x_grad,1)
                    h_grad = copy.deepcopy(OrderedDict(self.model.decoder.layer.named_parameters())["weight_hh_l0"].grad)
                    ref_grade[1].update(h_grad,1)
                    self.optim.step()  

                print("loss:",loss)
                print("grad_loss",grade_loss)
                print("loss_vae",loss_vae)
                loss_list.append(loss.tolist())
                loss_grad_list.append(grade_loss.tolist())
                loss_recon_list.append(loss_vae.tolist())
            writer.add_scalar('loss',np.mean(loss_list),epoch)
            writer.add_scalar('loss_recon',np.mean(loss_recon_list),epoch)
            writer.add_scalar('loss_grad',np.mean(loss_grad_list),epoch)

       
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """

        if self.train_mode != 0 and self.train_mode != 1:
            output_path = file_path + "train.task%d.ep%d" % (self.train_mode, epoch)
            torch.save(self.model.module.state_dict(), output_path)
            print("EP:%d Model Saved on:" % epoch, output_path)

            output_path = file_path + "encoder.task%d.ep%d" % (self.train_mode, epoch)
            torch.save(self.model.encoder.state_dict(), output_path)
            print("EP:%d Encoder Saved on:" % epoch, output_path)
            return output_path

        elif self.train_mode == 0 or self.train_mode == 1:
            output_path = file_path + "train.task%d.ep%d" % (self.train_mode, epoch)
            torch.save(self.model.state_dict(), output_path)
            print("EP:%d Model Saved on:" % epoch, output_path)

            output_path = file_path + "encoder.task%d.ep%d" % (self.train_mode, epoch)
            torch.save(self.model.encoder.state_dict(), output_path)
            print("EP:%d Encoder Saved on:" % epoch, output_path)
            return output_path

             
        
    def plot_traj(self,data, sequence_len,filename):
        #load_data_fea_label()
        from matplotlib import pyplot as plt
        x_list = []
        y_list = []
        for locatintrajid in range(0,sequence_len):
            x_list.append(data[locatintrajid][0].cpu().detach().numpy().tolist())
            y_list.append(data[locatintrajid][1].cpu().detach().numpy().tolist())
        plt.figure(figsize=(10, 7))
        per_seg_x_array = np.array(x_list)
        per_seg_y_array = np.array(y_list)
        plt.quiver(per_seg_x_array[:-1], per_seg_y_array[:-1], per_seg_x_array[1:]-per_seg_x_array[:-1], per_seg_y_array[1:]-per_seg_y_array[:-1], scale_units='xy', angles='xy', scale=1, headlength = 3, headaxislength = 3, headwidth = 3,color = "b", width = 0.005)
        plt.title(per_seg_x_array.size)
        plt.legend()
        plt.grid()
        plt.savefig("./recon_plot/"+filename[0:-4]+"png")
        np.savetxt('./reconstruct/reconstructor{}.txt'.format(filename),data.cpu().detach().numpy().tolist(),fmt='%s')
    