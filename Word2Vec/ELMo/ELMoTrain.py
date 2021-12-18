from ..utils.utils import get_loader,PAD_TOKEN
from ..ELMo.BiLMModel import BiLM
from ..ELMo.BiLMDataset import load_corpus,BiLMDataset
from ..ELMo.Config import Config
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy

def train():
    vocab_w , corpus_w, vocab_c, corpus_c = load_corpus(Config["train_file"])
    train_data = BiLMDataset(vocab_w , corpus_w, vocab_c, corpus_c)
    train_loader = get_loader()

    criterion = nn.CrossEntropyLoss(ignore_index=vocab_w[PAD_TOKEN], reduction="sum")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLM(Config, vocab_w, vocab_c)
    model = model.to(device)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           lr= Config["learning_rate"])

    model.train()

    for epoch in range(Config["num_epoch"]):
        total_loss = 0
        total_tags = 0
        for batch in tqdm(train_loader,desc=f"train epoch {epoch}"):
            batch = [x.to(device) for x in batch]
            input_w, input_c, seq_len, target_fw, target_bw = batch

            optimizer.zero_grad()
            outputs_fw, outputs_bw = model(input_c, seq_len)

            loss_fw = criterion(outputs_fw.view(-1, outputs_fw.shape[-1]),
                                target_fw.view(-1))

            loss_bw = criterion(outputs_bw.view(-1, outputs_bw.shape[-1]),
                                target_bw.view(-1))

            loss = (loss_bw+loss_fw)/2
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), Config['clip_grad'])
            optimizer.step()

            total_loss+=loss_fw.item()
            total_tags+=seq_len.sum().item()

            train_ppl = numpy.exp(total_loss/total_tags)
            print(f"train PPL {train_ppl}")
        model.save_pretrained(Config["model_path"])


