from torch.utils.data import Dataset
from tqdm import tqdm

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<Pad>"

class SkipGramDataset(Dataset):
    def __init__(self,corpus, vocab, context_size=2):
        self.data = []
        self.bos = BOS_TOKEN
        self.eos = BOS_TOKEN
        for sentence in tqdm(corpus,desc = "Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]

            for i in range(1,len(sentence)-1):
                left_context_idx = max(0,i-context_size)
                right_context_idx = min(len(sentence),i+context_size)
                context = sentence[left_context_idx:i] + sentence[i+1 : right_context_idx+1]
                target = sentence[i]
                self.data.extend([(target,c) for c in context])