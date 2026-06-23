from globals import *
from tokenizer import tokenize
import json
class Dataset:
    def __init__(self, dataset_path, start, size_data):
        with open (dataset_path, 'r') as f:
            self.dataset = json.load(f)
        # print(self.dataset)
        # print(self.dataset)
        # print(type(self.dataset))
        # self.dataset = tokenize(self.dataset)
        self.dataset =  self.dataset[start:min(len(self.dataset),start+size_data)]
        self.inp_enc = []
        
        #####################
        self.inp_dec = []
        self.gt = []
        #####################
        for obj in self.dataset:
            tokens = [SOS] + tokenize(obj["gt"]) + [EOS]
            inp_start_ind = 0
            inp_end_ind = len(tokens) - 2
            num_inp_toks = inp_end_ind - inp_start_ind + 1            
            num_pad_toks = num_toks - num_inp_toks
            self.inp_dec.append(tokens[inp_start_ind:inp_end_ind+1] + num_pad_toks*[PAD])
            self.gt.append(tokens[inp_start_ind+1:inp_end_ind+2] + num_pad_toks*[PAD])
            
            self.inp_enc.append('/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/' + obj['imagePath'])

        self.dataset_size = len(self.inp_enc)
        # print(self.inp[0])
        # print("==========================================")
        # print(self.gt[0])    
        
        # print("\n\n==========================================")
        
    def __getitem__(self, key):
        return self.inp_enc[key], self.inp_dec[key], self.gt[key]
    
    def getSize(self):
        return self.dataset_size
