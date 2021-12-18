import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import os



class BiLM(nn.Module):
    def __init__(self, configs, vocab_w, vocab_c):
        super(BiLM, self).__init__()

        self.dropout_prob = configs["dropout_prob"]
        self.num_classes = len(vocab_w)

        self.token_embedding_layer = ConvTokenEmbedLayer(
            vocab_c = vocab_c,
            char_embedding_dim = configs["char_embedding_dim"],
            char_conv_filter = configs["char_conv_filter"],
            num_highway = configs["num_highways"],
            output_dim = configs["projection_dim"],
        )

        self.lstm_encode_layer = ELMoLstmEncodeLayer(
            input_dim=configs["projection_dim"],
            hidden_dim= configs["hidden_dim"],
            num_layers = configs["num_layers"]
        )

        self.classifier = nn.Linear(configs["projection_dim"], self.num_classes)

    def forward(self,inputs, lengths):
        token_embedding = self.token_embedding_layer(inputs)
        token_embedding = F.dropout(token_embedding, self.dropout_prob)
        forword_stats, backward_stats = self.lstm_encode_layer(token_embedding, lengths)
        return self.classifier(forword_stats[-1]), self.classifier(backward_stats[-1])

    def save_pretrained(self,path):
        self.token_embedding_layer.load_state_dict(torch.load(os.path.join(path,"token_embedding.pth")))
        self.lstm_encode_layer.load_state_dict(torch.load(os.path.join(path,"encoder,pth")))



class ELMoLstmEncodeLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ELMoLstmEncodeLayer, self).__init__()

        self.projection_dim = input_dim
        self.num_layers = num_layers

        self.forward_layer = nn.ModuleList()
        self.backward_layer = nn.ModuleList()

        self.forward_projection = nn.ModuleList()
        self.backward_projection = nn.ModuleList()

        lstm_input_dim = input_dim
        for _ in range(num_layers):
            forward_layer = nn.LSTM(lstm_input_dim,hidden_dim,num_layers=1,batch_first=True)
            forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            backward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
            backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            lstm_input_dim = self.projection_dim

            self.forward_layer.append(forward_layer)
            self.forward_projection.append(forward_projection)

            self.backward_layer.append(backward_layer)
            self.backward_projection.append(backward_projection)

    def forward(self, inputs, lengths):
        batch_size, seq_len, input_dim = inputs.shape

        rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size,1)
        for i in range(lengths.shape[0]):
            rev_idx[:,:lengths[i]] = torch.arange(lengths[i]-1,-1,-1)

        rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
        rev_idx = rev_idx.to(inputs.device)
        rev_inputs = inputs.gather(1,rev_idx)

        forward_inputs, backward_inputs = inputs, rev_inputs
        stack_forward_states, stack_backward_states = [], []

        for idx_layer in range(self.num_layers):
            packed_forward_inputs = pack_padded_sequence(forward_inputs, lengths, batch_first=True, enforce_sorted=False)
            packed_backward_inputs = pack_padded_sequence(backward_inputs, lengths, batch_first=True, enforce_sorted=False)

            forward_layer = self.forward_layer[idx_layer]
            packed_forward, _ = forward_layer(packed_forward_inputs)
            forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
            forward = self.forward_projection[idx_layer](forward)
            stack_forward_states.append(forward)

            backward_layer = self.backward_layer[idx_layer]
            packed_backward, _ = backward_layer(packed_backward_inputs)
            backward = pad_packed_sequence(packed_backward, batch_first= True)[0]
            backward = self.backward_projection[idx_layer](backward)
            stack_backward_states.append(backward)

        return stack_forward_states, stack_backward_states

class ConvTokenEmbedLayer(nn.Module):
    def __init__(self, vocab_c, char_embedding_dim, char_conv_filter, num_highway, output_dim, pad="<pad>"):
        super(ConvTokenEmbedLayer, self).__init__()

        self.vocab_c = vocab_c
        self.char_embedding_layer = nn.Embedding(
            len(vocab_c),
            char_embedding_dim,
            padding_idx=vocab_c[pad]
        )

        self.char_embedding_layer.data.uniform(-0.25, 0.25)

        self.conv_layers = nn.ModuleList()
        for kernel_size, out_channels in char_conv_filter:
            conv = nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=True
            )
            self.conv_layers.append(conv)

        self.num_filter = sum([c[1] for c in char_conv_filter])
        self.num_highway = num_highway
        self.highway_layer = Highway(self.num_filter, self.num_highway, activation=F.relu)

        self.projection = nn.Linear(self.num_filter, output_dim, bias=True)

    def forward(self, inputs):
        batch_size, seq_len, token_len = inputs.shape
        # (batch_size*seq_len,token_len)
        inputs = inputs.view(batch_size * seq_len, -1)
        # (batch_size*seq_len,token_len,char_embedding_dim)
        char_embedding = self.char_embedding_layer(inputs)
        # (batch_size*seq_len,char_embedding_dim,token_len)
        char_embedding = char_embedding.transpose(1, 2)


        conv_hiddens = []
        for i in range(len(self.conv_layers)):
            # (batch_size*seq_len,out_channels,token_len^)
            conv_hidden = self.conv_layers[i](char_embedding)
            # (batch_size*seq_len,out_channels)
            conv_hidden, _ = torch.max(conv_hidden,-1)
            conv_hidden = F.relu(conv_hidden)
            conv_hiddens.append(conv_hidden)

        # (batch_size*seq_len,out_channels*num_filters)
        token_embeding = torch.cat(conv_hiddens,-1)
        # (batch_size*seq_len,out_channels*num_filters)
        token_embeding = self.highway_layer(token_embeding)
        # (batch_size*seq_len,output_dim)
        token_embeding = self.projection(token_embeding)
        token_embeding = token_embeding.view(batch_size,seq_len,-1)
        return token_embeding


class Highway(nn.Module):
    def __init__(self, input_dim, num_layers, activation=F.relu):
        super(Highway, self).__init__()

        self.input_dim = input_dim
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim * 2) for _ in num_layers])

        for layer in self.layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        curr_input = inputs
        for layer in self.layers:
            projected_input = layer(curr_input)
            hidden = self.activation(projected_input[:, :self.input_dim])
            gate = F.sigmoid(projected_input[:, self.input_dim:])
            curr_input = gate * curr_input + (1 - gate) * hidden
        return curr_input
