import torch
from torch import nn
from torch import Tensor
import numpy as np


class Alastor (nn.Module):
    def __init__(self, n_channel=4, max_seq_len=2000, num_targets=4+1,
                conv1kc=32, conv1ks=15, conv1st=1, conv1pd=7, pool1ks=8, pdrop1=0.5, #conv_block_1 parameters
                conv2kc=48, conv2ks=3, conv2st=1, conv2pd=3, pool2ks=4 , pdrop2=0.5, #conv_block_2 parameters
                conv3kc=32, conv3ks=3, conv3st=1, conv3pd=3, pool3ks=4, pdrop3=0.4, #conv_block_3 parameters
                convdc=7, convd1kc=32 , convd1ks=3, convd2kc=96 , convd2ks=1, pdropd=0.3,
                conv4kc=64, conv4ks=1, conv4st=1, conv4pd=0, pdrop4=0.5):
        super(Alastor, self).__init__()

        self.convdc = convdc
        self.device = 'cpu'
        self.max_seq_len = max_seq_len
        self.num_targets = num_targets

        conv1pd = int((conv1ks - 1) / 2)
        conv2pd = int((conv2ks - 1) / 2)
        conv3pd = int((conv3ks - 1) / 2)

        convd2kc = int(conv3kc)
        
        self.conv_block_1 = nn.Sequential(
            #nn.GELU(),
            nn.Conv1d(n_channel, conv1kc, kernel_size=conv1ks, stride=conv1st, padding=conv1pd, bias=False),
            nn.BatchNorm1d(conv1kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool1ks),
            nn.Dropout(p=pdrop1))
                
        self.conv_block_2 = nn.Sequential(
           # nn.GELU(),
            nn.Conv1d(conv1kc, conv2kc, kernel_size=conv2ks, stride=conv2st, padding=conv2pd, bias=False),
            nn.BatchNorm1d(conv2kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool2ks),
            nn.Dropout(p=pdrop2))
        
        self.conv_block_3 = nn.Sequential(
           # nn.GELU(),
            nn.Conv1d(conv2kc, conv3kc, kernel_size=conv3ks, stride=conv3st, padding=conv3pd, bias=False),
            nn.BatchNorm1d(conv3kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool3ks),
            nn.Dropout(p=pdrop3))
        
        self.dilations = nn.ModuleList()
        
        for i in range(convdc):
            padding = 2**(i)
            self.dilations.append(nn.Sequential(
              #  nn.GELU(),
                nn.Conv1d(conv3kc, convd1kc, kernel_size=convd1ks, padding=padding, dilation=2**i, bias=False),
                nn.BatchNorm1d(convd1kc, momentum=0.9, affine=True), 
                nn.GELU(),
                nn.Conv1d(convd1kc, convd2kc, kernel_size=convd2ks, padding=0, bias=False),
                nn.BatchNorm1d(convd2kc, momentum=0.9, affine=True), 
                nn.Dropout(p=pdropd)))
            
        self.conv_block_4 = nn.Sequential(
           # nn.GELU(),
            nn.Conv1d(convd2kc, conv4kc, kernel_size=conv4ks, stride=conv4st, padding=conv4pd, bias=False),
            nn.BatchNorm1d(conv4kc, momentum=0.9, affine=True), 
            nn.Dropout(p=pdrop4)) 
        
        self.conv_block_5 = nn.Sequential(
            #nn.GELU(),
            nn.Linear(conv4kc, 1, bias=True),
            nn.Flatten())
        
        self.mu_block = nn.Sequential(
            nn.Linear(int(self.max_seq_len / (2**7)), 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )
        
    def forward (self, seq):
        
        out1 = self.conv_block_1(seq)
        out1 = self.conv_block_2(out1)
        out1 = self.conv_block_3(out1)
        current = out1
        for i in range(self.convdc):
            residual = self.dilations[i](current)
            current = current.add(residual)
        
        out = self.conv_block_4(current)
        out = out.transpose(1, 2)
        out = self.conv_block_5(out)
        out = self.mu_block (out)
        return (ou                                                                 t)
        
    def compile (self, device = 'cpu'):
        self.to(device)
        self.device = device

class Alastor_Beta (nn.Module):
    def __init__(self, n_channel=4, max_seq_len=2000, num_targets=4+1,
                conv1kc=64, conv1ks=13, conv1st=1, conv1pd=7, pool1ks=8, pdrop1=0.5, #conv_block_1 parameters
                conv2kc=48, conv2ks=7, conv2st=1, conv2pd=3, pool2ks=4 , pdrop2=0.4, #conv_block_2 parameters
                conv3kc=32, conv3ks=15, conv3st=1, conv3pd=3, pool3ks=4, pdrop3=0.2, #conv_block_3 parameters
                convdc=5, convd1kc=32 , convd1ks=3, convd2kc=96 , convd2ks=1, pdropd=0,
                conv4kc=32, conv4ks=1, conv4st=1, conv4pd=0, pdrop4=0.2):
        super(Alastor, self).__init__()

        self.convdc = convdc
        self.device = 'cpu'
        self.max_seq_len = max_seq_len
        self.num_targets = num_targets

        conv1pd = int((conv1ks - 1) / 2)
        conv2pd = int((conv2ks - 1) / 2)
        conv3pd = int((conv3ks - 1) / 2)

        convd2kc = int(conv3kc)
        
        self.conv_block_1 = nn.Sequential(
            nn.Tanhshrink(),
            nn.Conv1d(n_channel, conv1kc, kernel_size=conv1ks, stride=conv1st, padding=conv1pd, bias=False),
            nn.BatchNorm1d(conv1kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool1ks),
            nn.Dropout(p=pdrop1))
                
        self.conv_block_2 = nn.Sequential(
            nn.Tanhshrink(),
            nn.Conv1d(conv1kc, conv2kc, kernel_size=conv2ks, stride=conv2st, padding=conv2pd, bias=False),
            nn.BatchNorm1d(conv2kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool2ks),
            nn.Dropout(p=pdrop2))
        
        self.conv_block_3 = nn.Sequential(
            nn.Tanhshrink(),
            nn.Conv1d(conv2kc, conv3kc, kernel_size=conv3ks, stride=conv3st, padding=conv3pd, bias=False),
            nn.BatchNorm1d(conv3kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool3ks),
            nn.Dropout(p=pdrop3))
        
        self.dilations = nn.ModuleList()
        
        for i in range(convdc):
            padding = 2**(i)
            self.dilations.append(nn.Sequential(
                nn.Tanhshrink(),
                nn.Conv1d(conv3kc, convd1kc, kernel_size=convd1ks, padding=padding, dilation=2**i, bias=False),
                nn.BatchNorm1d(convd1kc, momentum=0.9, affine=True), 
                nn.Tanhshrink(),
                nn.Conv1d(convd1kc, convd2kc, kernel_size=convd2ks, padding=0, bias=False),
                nn.BatchNorm1d(convd2kc, momentum=0.9, affine=True), 
                nn.Dropout(p=pdropd)))
            
        self.conv_block_4 = nn.Sequential(
            nn.Tanhshrink(),
            nn.Conv1d(convd2kc, conv4kc, kernel_size=conv4ks, stride=conv4st, padding=conv4pd, bias=False),
            nn.BatchNorm1d(conv4kc, momentum=0.9, affine=True), 
            nn.Dropout(p=pdrop4)) 
        
        self.conv_block_5 = nn.Sequential(
            nn.Tanhshrink(),
            nn.Linear(conv4kc, 1, bias=True),
            nn.Flatten())
        
        self.mu_block = nn.Sequential(
            nn.Linear(int(self.max_seq_len / (2**7)), 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )
        
    def forward (self, seq):
        
        out1 = self.conv_block_1(seq)
        out1 = self.conv_block_2(out1)
        out1 = self.conv_block_3(out1)
        current = out1
        for i in range(self.convdc):
            residual = self.dilations[i](current)
            current = current.add(residual)
        
        out = self.conv_block_4(current)
        out = out.transpose(1, 2)
        out = self.conv_block_5(out)
        out = self.mu_block (out)
        return (out)
        
    def compile (self, device = 'cpu'):
        self.to(device)
        self.device = device
        
def ConvBlock (in_channels, out_channels, pool=False):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm1d(out_channels)]
    if pool:
        layers.append(nn.MaxPool1d(4))
    return nn.Sequential(*layers)

class Filius (nn.Module):
    def __init__ (self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock (64, 128, pool= True)
        self.res1 = nn.Sequential (ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock (128, 256, pool= True)
        self.conv4 = ConvBlock (256, 512, pool= True)
        self.res2 = nn.Sequential (ConvBlock(512, 512), ConvBlock(512, 512))
        self.conv5 = ConvBlock (512, 512, pool= True)
        self.classifier = nn.Sequential (nn.MaxPool1d(4), nn.Flatten(), nn.Linear(512,1))
        self.dilations = nn.ModuleList()
        
        # for i in range(5):
        #     padding = 2**(i)
        #     self.dilations.append(nn.Sequential(
        #         nn.Conv1d(512, 64, kernel_size=3,  dilation=2**i, bias=False),
        #         nn.BatchNorm1d(64, momentum=0.9, affine=True), 
        #         nn.Conv1d(64, 96, kernel_size=1, padding=0, bias=False),
        #         nn.BatchNorm1d(96, momentum=0.9, affine=True), 
        #         nn.Dropout(p=0.2)))
    def resnet_block (self, xb):
        out_=[]
        for i in range(self.num_classes):
            out = self.conv1 (xb)
            out = self.conv2 (out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out)+ out
            out = self.conv5(out) 
            out = self.classifier(out)
            out_.append(out)
        final_out= torch.transpose(torch.stack(out_),0,1)
        return final_out
    def forward (self, xb):
        out = self.resnet_block(xb)
        return (out)    