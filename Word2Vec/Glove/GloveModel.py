import torch.nn as nn
import torch

class GloveModel(nn.Module):
    def __init__(self,vocab_size, embeding_dim):
        super(GloveModel,self).__init__()
        self.w_embedding = nn.Embedding(vocab_size,embeding_dim)
        self.w_bias = nn.Embedding(vocab_size,1)

        self.c_embedding = nn.Embedding(vocab_size, embeding_dim)
        self.c_bias = nn.Embedding(vocab_size,1)


    def forward_w(self,words):
        w_embeds = self.w_embedding(words)
        w_bias = self.w_bias(words)
        return w_embeds,w_bias

    def forward_c(self,contexts):
        c_embeds = self.c_embedding(contexts)
        c_bias = self.c_bias(contexts)
        return c_embeds,c_bias

