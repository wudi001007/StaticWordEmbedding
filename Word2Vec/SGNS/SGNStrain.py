import torch
import torch.optim as optim
from ..utils.utils import load_reuters,get_loader,save_pretrained
from ..SGNS.SGNSDataset import SGNSDataset
from ..SGNS.SGNSModel import SGNSModel
from tqdm import tqdm
import torch.nn.functional as F

def get_unigram_distribution(corpus,vocab_size):
    token_counts = torch.tensor([0]*vocab_size)
    total_count = 0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_counts[token] += 1
    unigram_dist = torch.div(token_counts.float(), total_count)
    return unigram_dist



def train():
    embedding_dim = 8
    context_size = 2
    hidden_dim =128
    batch_size = 1
    num_epoch = 10
    n_negative = 10

    corpus, vocab = load_reuters()
    unigram_dist = get_unigram_distribution(corpus,len(vocab))
    negative_sampling_dist = unigram_dist**0.75
    negative_sampling_dist /= negative_sampling_dist.sum()

    dataset = SGNSDataset(
        corpus,
        vocab,
        context_size=context_size,
        n_negative=n_negative,
        ns_dist=negative_sampling_dist
    )

    data_loader = get_loader(dataset,batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SGNSModel(len(vocab),embedding_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader,desc=f"Training Epoch {epoch}"):
            words, contexts ,neg_contexts = [x.to(device) for x in batch]
            optimizer.zero_grad()
            batch_size=words.shape[0]

            word_embeds = model.forward_w(words).unsqueeze(dim=2)
            context_embeds = model.forward_c(contexts)
            neg_context_embeds = model.forward_c(neg_contexts)

            context_loss = F.logsigmoid(torch.bmm(context_embeds,word_embeds).squeeze(dim=2))
            context_loss = context_loss.mean(dim=1)

            neg_context_loss = F.logsigmoid(torch.bmm(neg_context_embeds,word_embeds).squeeze(dim=2).neg())
            neg_context_loss = neg_context_loss.view(batch_size,-1,n_negative)
            neg_context_loss = neg_context_loss.sum(dim=2)
            neg_context_loss = neg_context_loss.mean(dim=1)

            loss = -(context_loss+neg_context_loss).mean()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

        print(f"Loss: {total_loss:.2f}")
    combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight
    save_pretrained(vocab, combined_embeds.data, "sgns.vec")

