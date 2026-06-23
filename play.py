from enc_dec import EncDec
from dataset import Dataset
from trainer import Trainer
from globals import *
from inference import Inference
import torch
model = EncDec(512, 16).to(device='cuda')
# model.load_state_dict(torch.load('model.pth', weights_only=True))
train_dataset = Dataset('train.json', 0, 1000)

trainer = Trainer(model, train_dataset).to(device='cuda')

trainer.start_training(16, 3, 1e-4)    #BATCH EPOCH LR
torch.save(model.state_dict(), './model.pth')

inference = Inference(model)
base = "/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images"
print(inference.generate(base + '/001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg'))



