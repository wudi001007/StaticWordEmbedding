from torch import nn
import torch.nn.functional as F

class SGNSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SGNSModel, self).__init__()
        self.w_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.c_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward_w(self,words):

        w_embeds = self.w_embedding(words)
        return w_embeds

    def forward_c(self,contexts):

        c_embeds = self.c_embedding(contexts)
        return c_embeds