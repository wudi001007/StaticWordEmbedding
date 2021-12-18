import torch
from torch import nn
import torch.nn.functional as F

class CbowModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CbowModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.classify = nn.Linear(embedding_dim,vocab_size,bias = False)

    def forward(self,inputs):

        embed = self.embedding_layer(inputs)
        hidden = embed.mean(dim=1)
        output = self.classify(hidden)
        logit =F.log_softmax(output, dim=1)
        return logit

