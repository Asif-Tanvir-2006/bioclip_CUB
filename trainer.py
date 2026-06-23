import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from globals import vocab_size
# import torch.nn.functional as F
from globals import PAD
class Trainer(nn.Module):
    def __init__(self, model, train_dataset, eval_dataset=None):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    def start_training(self, batch_size, epochs, lr):
        optimiser = optim.AdamW(self.model.parameters(), lr=lr)
        loss = None
        num_batches = self.train_dataset.getSize() / batch_size
        for i in range(1, epochs+1):
            print(i)
            start_i = 0
            cum_loss = 0
            while(start_i < self.train_dataset.getSize()):
                optimiser.zero_grad()
                end_i = min(self.train_dataset.getSize()-1, start_i+batch_size-1)
                inp_enc, inp_dec, gt = self.train_dataset[start_i:end_i+1]
                inp_dec = torch.tensor(inp_dec, dtype=torch.long, requires_grad=False, device='cuda')
                gt = torch.tensor(gt, dtype=torch.long, requires_grad=False, device='cuda')

                pred = self.model(inp_enc, inp_dec)
                loss = self.loss_fn(pred.view(-1, vocab_size), gt.view(-1))
                cum_loss += loss.item()
                loss.backward()
                optimiser.step()
                start_i = end_i + 1
                
            if(loss is not None):
                print("Train: ", cum_loss/num_batches)
            


            