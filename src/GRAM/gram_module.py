#################################################################
# Code written by Xueping Peng according to GRAM paper and original codes
#################################################################
import torch
import torch.nn as nn
import numpy as np


class DAGAttention(nn.Module):
    def __init__(self, in_features, attention_dim_size):
        super(DAGAttention, self).__init__()
        self.attention_dim_size = attention_dim_size
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, attention_dim_size)
        self.linear2 = nn.Linear(attention_dim_size, 1)

    def forward(self, leaves, ancestors):
        # concatenate the leaves and ancestors
        x = torch.cat((leaves, ancestors), dim=-1)
        # Linear layer
        x = self.linear1(x)
        # tanh activation
        x = torch.tanh(x)
        # linear layer
        x = self.linear2(x)
        # softmax activation
        x = torch.softmax(x, dim=0)
        x = x.reshape(x.size()[1], x.size()[0])
        # weighted sum on ancestors
        x = x.matmul(ancestors).sum(dim=0)

        return x


class DAGAttention2D(nn.Module):
    def __init__(self, in_features, attention_dim_size):
        super(DAGAttention2D, self).__init__()
        self.attention_dim_size = attention_dim_size
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, attention_dim_size)
        self.linear2 = nn.Linear(attention_dim_size, 1)

    def forward(self, leaves, ancestors, mask=None):
        # concatenate the leaves and ancestors
        x = torch.cat((leaves, ancestors), dim=-1)

        # Linear layer
        x = self.linear1(x)
        # tanh activation
        x = torch.tanh(x)
        # linear layer
        x = self.linear2(x)

        if mask is not None:
            mask = mask.unsqueeze(2)
            mask = (1.0 - mask) * -10000.0
            x = x + mask
        # softmax activation
        x = torch.softmax(x, dim=1)

        # weighted sum on ancestors
        x = (x * ancestors).sum(dim=1)
        return x


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h, mask, lengths):
        out, h = self.gru(x, h)
        out = self.fc(out)

        out = torch.sigmoid(out)
        # out = torch.softmax(out, -1)
        mask_t = torch.from_numpy(mask).to(self.device)
        mask_t = mask_t.reshape((mask_t.size(0), mask_t.size(1), 1))
        out = out * mask_t

        # get last valid output
        batch_size, len_seqs, dim_size = out.shape[0], out.shape[1], out.shape[2]
        out_reshape = out.view(-1, out.shape[-1])
        index_last_valid_output = [i * len_seqs + lengths[i] - 1 for i in range(batch_size)]
        last_valid_output = out_reshape[index_last_valid_output]

        return out, h, last_valid_output

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(out[:, -1])
        # out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


class GRAM(nn.Module):

    def __init__(self, leaves_list, ancestors_list, leaf_codes_num, internal_codes_num, emb_dim_size,
                 attn_dim_size, rnn_dim_size, num_class, device):
        super(GRAM, self).__init__()
        self.leaf_codes_num = leaf_codes_num
        self.internal_codes_num = internal_codes_num
        self.emb_dim_size = emb_dim_size
        self.attn_dim_size = attn_dim_size
        self.rnn_dim_size = rnn_dim_size
        self.num_class = num_class
        self.device = device

        # embedding for leaf codes and internal codes
        self.embed_init = nn.Embedding(leaf_codes_num + internal_codes_num, emb_dim_size).to(device)
        # DAG attention
        self.dag_attention = DAGAttention(2*emb_dim_size, attn_dim_size).to(device)
        # GRU network
        self.gru = GRUNet(attn_dim_size, rnn_dim_size, num_class, 2, device)

        # Calculate the DAG Attention
        attn_embed_list = []
        for ls, ans in zip(leaves_list, ancestors_list):
            ls = np.array(ls).astype(np.long)
            ans = np.array(ans).astype(np.long)

            leaves_emb = self.embed_init(torch.from_numpy(ls).to(device))
            ancestors_emb = self.embed_init(torch.from_numpy(ans).to(device))

            attn_embd = self.dag_attention(leaves_emb, ancestors_emb)
            attn_embed_list.append(attn_embd)

        dag_emb = torch.cat(attn_embed_list, dim=-1)
        self.dag_emb = dag_emb.reshape((int(dag_emb.size()[0] / self.attn_dim_size), self.attn_dim_size))

    def forward(self, inputs, mask, lengths):

        x = torch.tanh(torch.matmul(inputs, self.dag_emb))
        h = self.gru.init_hidden(mask.shape[0]).data
        out, h, valid_output = self.gru(x.to(self.device).float(), h, mask, lengths)

        return out, valid_output





