import torch
from tokenizer import tokenize, detokenize
from globals import *

class Inference:
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Ensure the model is in evaluation mode
    
    def generate(self, inp, max_length=1000):
        ans = [SOS]
        with torch.inference_mode():
            enc_out = self.model.encoder([inp])
            while len(ans) < max_length:
                inp_dec = torch.tensor(
                    [ans[-num_toks:]],
                    dtype=torch.long,
                    device='cuda'
                )
                logits = self.model.decoder(
                    inp_dec,
                    enc_out
                )
                token_id = logits[0, -1].argmax().item()
                if token_id == EOS:
                    break
                ans.append(token_id)
        return detokenize(ans[2:])
