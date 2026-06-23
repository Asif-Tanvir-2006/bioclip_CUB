import torch.nn as nn
from bioclip_enc import BioClip
from decoder import Decoder
from globals import *
class EncDec(nn.Module):
    def __init__(self, embed_dim, num_attn_heads):
        super().__init__()
        self.encoder = BioClip()
        self.decoder = Decoder(embed_dim=embed_dim, num_attn_heads=num_attn_heads, vocab_size=vocab_size)
        
         
    
    def forward(self, inp_enc, inp_dec):
        enc_out = self.encoder(inp_enc) 
        dec_out = self.decoder(inp_dec, enc_out)
        return dec_out
        