from torch.utils.data import Dataset
from tqdm import tqdm

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<Pad>"

class CbowDataset(Dataset):
    def __init__(self,corpus, vocab, context_size=2):
        self.data = []
        self.bos = BOS_TOKEN
        self.eos = BOS_TOKEN
        for sentence in tqdm(corpus,desc = "Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size*2+1:
                continue

            for i in range(context_size, len(sentence)-context_size):
                context = sentence[i-context_size:i] + sentence[i+1:i+context_size+1]
                target = sentence[i]
                self.data.append((target,context))

