import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import *
#block_size = token_len
# class GlobalVars():
#     def __init__(self):
#         self.pad_mask = None
        
# globalVars = GlobalVars()
device = 'cuda'
class SelfAttention(nn.Module):
    def __init__(self, head_size, embed_dim):
        super().__init__()
        self.head_size = head_size
        self.key_weight = nn.Linear(embed_dim, self.head_size, bias=False)
        self.query_weight = nn.Linear(embed_dim, self.head_size, bias=False)
        self.value_weight = nn.Linear(embed_dim, self.head_size, bias=False)
    def forward(self, inp, pad_mask):
        _, T, _ = inp.shape
        key = self.key_weight(inp)
        query = self.query_weight(inp)
        value = self.value_weight(inp) #(B, T, D)
        attention_scores = (query @ torch.transpose(key, -2, -1)) / ((self.head_size)**0.5)
        mask = torch.tril(torch.ones([T,T])).to(device=device)
        # print(globalVars.pad_mask)
        masked_attention_scores = torch.masked_fill(attention_scores, mask==0, float('-inf'))  #(B T T)
        masked_attention_scores = torch.masked_fill(masked_attention_scores, pad_mask==True, float('-inf'))  #(B T T)
        
        normalised_masked_attention_scores = torch.softmax(masked_attention_scores, dim=-1)
        return normalised_masked_attention_scores @ value
class CrossAttention(nn.Module):
    def __init__(self, head_size, embed_dim):
        super().__init__()
        self.head_size = head_size
        self.key_weight = nn.Linear(embed_dim, self.head_size, bias=False)
        self.query_weight = nn.Linear(embed_dim, self.head_size, bias=False)
        self.value_weight = nn.Linear(embed_dim, self.head_size, bias=False)
    def forward(self, inp, enc_out):
        _, T, _ = inp.shape
        key = self.key_weight(enc_out)
        query = self.query_weight(inp)
        value = self.value_weight(enc_out) #(B, T, D)
        attention_scores = (query @ torch.transpose(key, -2, -1)) / ((self.head_size)**0.5)
        # mask = torch.tril(torch.ones([T,T])).to(device=device)
        # print(globalVars.pad_mask)
        # masked_attention_scores = torch.masked_fill(attention_scores, mask==0, float('-inf'))  #(B T T)
        # masked_attention_scores = torch.masked_fill(attention_scores, enc_pad_mask==True, float('-inf'))  #(B T T)
        
        normalised_masked_attention_scores = torch.softmax(attention_scores, dim=-1)
        return normalised_masked_attention_scores @ value

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([SelfAttention(head_size=self.head_size, embed_dim=embed_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, pad_mask):
        return self.proj(torch.cat([head(x, pad_mask) for head in self.heads], dim=-1))

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([CrossAttention(head_size=self.head_size, embed_dim=embed_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, enc_out):
        return self.proj(torch.cat([head(x, enc_out) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.proj = nn.Linear(4*embed_dim, embed_dim)
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            self.proj
        )
    
    def forward(self, inp):
        return self.layer(inp)
        
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.ffl = FeedForward(embed_dim)
        self.self_attn = MultiHeadSelfAttention(num_heads, embed_dim)
        self.cross_attn = MultiHeadCrossAttention(num_heads, embed_dim)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        
    def forward(self, inp, enc_out, pad_mask):
        inp = inp + self.self_attn(self.ln1(inp),pad_mask)
        inp = inp + self.cross_attn(self.ln2(inp),enc_out)
        inp = inp + self.ffl(self.ln3(inp))
        return inp
    
class Decoder(nn.Module):
    def __init__(self, embed_dim, num_attn_heads, vocab_size):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # self.pad_mask = None
        
        self.token_embeddings = nn.Embedding(self.vocab_size, self.embed_dim) #C -> channels -> embed_dim
        self.pos_embeddings = nn.Embedding(num_toks, self.embed_dim)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_attn_heads),
            Block(embed_dim, num_attn_heads),
        ]            
        )
        self.layernorm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size)
        
        ##attention
                
    
        


    # def forward(self, inp):
    #     B, T = inp.shape
    #     pad_mask = (inp==PAD).unsqueeze(1)
    #     # pad_mask = pad_mask + torch.transpose(pad_mask, -2, -1)
    #     globalVars.pad_mask = pad_mask
    #     tok_emb = self.token_embeddings(inp) #(B T C)
    #     pos_emb = self.pos_embeddings(torch.arange(T).to(device=device)) #(T C)
    #     x = tok_emb + pos_emb  # (B T C) + ( _ T C)
    #     return (self.lm_head(self.layernorm(self.blocks(x))))
        
        
    ## THIS ONE IS FOR ENC-DEC     
    def forward(self, inp, enc_out):
        _, T = inp.shape
        pad_mask = (inp==PAD).unsqueeze(1)
        # enc_pad_mask, enc_out = enc_out
        tok_emb = self.token_embeddings(inp) #(B T C)
        pos_emb = self.pos_embeddings(torch.arange(T).to(device=device)) #(T C)
        x = tok_emb + pos_emb  # (B T C) + ( _ T C)
        for block in self.blocks:
            x = block(x, enc_out, pad_mask)
        return (self.lm_head(self.layernorm(x)))
        
    