from ..utils.utils import PAD_TOKEN,EOS_TOKEN,BOS_TOKEN,BOW_TOKEN,EOW_TOKEN
from ..utils.Vocab import Vocab
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

class BiLMDataset(Dataset):
    def __init__(self, vocab_w , corpus_w, vocab_c, corpus_c):
        super(BiLMDataset,self).__init__()
        self.pad_w = vocab_w[PAD_TOKEN]
        self.pad_c = vocab_c[PAD_TOKEN]

        self.data = []

        for sent_w, sent_c in tqdm(zip(corpus_w, corpus_c)):
            self.data.append((sent_w, sent_c))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, example):
        seq_len = torch.LongTensor([len(ex[0]) for ex in example])

        input_w = [torch.tensor(ex[0]) for ex in example]
        input_w = pad_sequence(input_w, batch_first=True, padding_value=self.pad_w)

        batch_size, max_seq_len = input_w.shape

        max_tok_len = max([max([len(token) for token in ex[1]]) for ex in example])
        input_c = torch.LongTensor(batch_size, max_seq_len, max_tok_len).fill_(self.pad_c)

        for i, (sent_w, sent_c) in enumerate(example):
            for j,tok in enumerate(sent_c):
                input_c[i][j][:len(tok)] = torch.LongTensor(tok)

        target_fw = torch.LongTensor(input_w.shape).fill_(self.pad_w)
        target_bw = torch.LongTensor(input_c.shape).fill_(self.pad_w)

        for i, (sent_w, sent_c) in enumerate(example):
            target_fw[i][:len(sent_w)-1] = torch.LongTensor(sent_w[1:])
            target_bw[i][1:len(sent_w)] = torch.LongTensor(sent_w[:len(sent_w)-1])

        return input_w, input_c, seq_len, target_fw, target_bw



def load_corpus(path, max_token_len, max_seq_len):
    text = []
    charset = {PAD_TOKEN, EOW_TOKEN, EOS_TOKEN, BOS_TOKEN, BOW_TOKEN}
    with open(path, "w", encoding="utf-8") as f:
        for line in tqdm(f):
            tokens = line.rstrip().split(" ")
            if max_seq_len is not None and len(tokens)+2>max_seq_len:
                tokens = tokens[:max_seq_len-2]
            sent=[BOS_TOKEN]
            for token in tokens:
                if max_token_len is not None and len(token)+2>max_token_len:
                    token = token[:max_token_len-2]
                    sent.append(token)
                    for char in token:
                        charset.add(char)
            sent.append(EOS_TOKEN)
            text.append(sent)

    vocab_w = Vocab.build(text,min_freq=2,reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    vocab_c = Vocab(tokens = list(charset))

    corpus_w = [vocab_w.convert_tokens_to_ids(sent) for sent in text]
    corpus_c = []

    bow = vocab_c[BOW_TOKEN]
    eow = vocab_c[EOS_TOKEN]

    for idx, sen in enumerate(text):
        sent_c = []
        for token in sen:
            if token == BOS_TOKEN or token == EOS_TOKEN:
                token_c = [bow, vocab_c[token], eow]
            else:
                token_c = [bow] + vocab_c.convert_tokens_to_ids(token) + [eow]
            sent_c.append(token_c)
        corpus_c.append(sent_c)

    return vocab_w , corpus_w, vocab_c, corpus_c




