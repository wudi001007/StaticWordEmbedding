from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from ..utils.utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from collections import defaultdict
class GloveDataset(Dataset):
    def __init__(self, corpus, vocab, context_size = 2):
        self.cooccur_counts = defaultdict(float)
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus,desc="dataset"):
            sentence = [self.bos]+sentence+[self.eos]
            for i in range(1,len(sentence)-1):
                w = sentence[i]
                left_context = sentence[max(0,i-context_size):i]
                right_context = sentence[i+1:min(i+context_size,len(sentence))+1]
                for k,c in enumerate(left_context[::-1]):
                    self.cooccur_counts[(w,c)] += 1/(k+1)
                for k,c in enumerate(right_context):
                    self.cooccur_counts[(w,c)] += 1/(k+1)

        self.data = [(w,c,count) for (w,c),count in self.cooccur_counts.items()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self,example):
        words = torch.tensor([ex[0] for ex in example])
        contexts = torch.tensor([ex[1] for ex in example])
        counts = torch.tensor([ex[2] for ex in example])


