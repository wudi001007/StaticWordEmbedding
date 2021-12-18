from torch import nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.classify = nn.Linear(embedding_dim,vocab_size,bias = False)

    def forward(self,inputs):

        embed = self.embedding_layer(inputs)
        output = self.classify(embed)
        logit =F.log_softmax(output, dim=1)
        return logit
