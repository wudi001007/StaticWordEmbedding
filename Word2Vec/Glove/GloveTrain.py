import torch
import torch.optim as optim
from ..utils.utils import load_reuters,get_loader,save_pretrained
from ..Glove.GloveModel import GloveModel
from ..Glove.GloveDataset import GloveDataset
from tqdm import tqdm

def train():
    embedding_dim = 64
    context_size = 2
    batch_size = 1024
    num_epoch = 10

    m_max =100
    alpha = 0.75

    corpus, vocab = load_reuters()
    dataset = GloveDataset(corpus,vocab,context_size= context_size)
    data_loader = get_loader(dataset,batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GloveModel

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc="Training Epoch {epoch}"):
            word,contexts,counts = [x.to(device) for x in batch]
            word_embed,word_bias = model.forward_w(word)
            context_embed,context_bias = model.forward_c(contexts)
            log_counts = torch.log(counts+1)

            weight_factor = torch.clamp(torch.pow(counts/m_max, alpha), max=1.0)
            optimizer.zero_grad()

            loss = (torch.sum(word_embed*context_embed,dim=1)+word_bias+context_bias-log_counts)**2
            wavg_loss = (weight_factor*loss).mean()
            wavg_loss.backward()
            optimizer.step()
            total_loss+=wavg_loss.item()
        print(f"Loss: {total_loss:.2f}")
    combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight
    save_pretrained(vocab, combined_embeds.data, "glove.vec")
