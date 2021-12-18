from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from ..utils.utils import BOS_TOKEN,EOS_TOKEN,PAD_TOKEN

class SGNSDataset(Dataset):
    def __init__(self,corpus, vocab, context_size=2,n_negative=5,ns_dist=None):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus,desc = "Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]


            for i in range(1,len(sentence)-1):
                left_context_idx = max(0,i-context_size)
                right_context_idx = min(len(sentence),i+context_size)
                context = sentence[left_context_idx:i] + sentence[i+1 : right_context_idx+1]
                target = sentence[i]

                context += [self.pad]*(context_size*2-len(context))
                self.data.append((target,context))

        self.n_negative = n_negative
        self.ns_dist = ns_dist if ns_dist is not None else torch.ones(len(vocab))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples],dtype=torch.long)
        contexts = torch.tensor([ex[1] for ex in examples],dtype=torch.long)
        batch_size, contexts_size = contexts.shape
        neg_context = []

        for i in range(batch_size):
            ns_dist = self.ns_dist.index_fill(0,contexts[i],.0)
            neg_context.append(torch.multinomial(ns_dist,self.n_negative*contexts_size,replacement=True))
        neg_context = torch.stack(neg_context,dim=0)
        return words,contexts,neg_context
