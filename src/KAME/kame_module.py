#################################################################
# Code written by Xueping Peng according to GRAM paper and original codes
#################################################################
import torch
import torch.nn as nn
import numpy as np
from GRAM.gram_module import DAGAttention, DAGAttention2D
# from src.GRAM.gram_module import DAGAttention

VERY_NEGATIVE_NUMBER = -1e30


class KAME(nn.Module):

    def __init__(self, leaves_list, ancestors_list, masks_list, internal_list, internal_ancestors_list,
                 masks_ancestors_list, leaf_codes_num,
                 internal_codes_num, emb_dim_size, attn_dim_size, rnn_dim_size, g_dim_size, num_class, device,
                 drop_prob=0.2):
        super(KAME, self).__init__()
        self.leaf_codes_num = leaf_codes_num
        self.internal_codes_num = internal_codes_num
        self.emb_dim_size = emb_dim_size
        self.attn_dim_size = attn_dim_size
        self.rnn_dim_size = rnn_dim_size
        self.g_dim_size = g_dim_size
        self.num_class = num_class
        self.device = device

        self.leaves_list = leaves_list
        self.ancestors_list = ancestors_list
        self.masks_list = masks_list
        self.internal_list = internal_list
        self.internal_ancestors_list = internal_ancestors_list
        self.masks_ancestors_list = masks_ancestors_list

        # embedding for leaf codes and internal codes
        self.linear = nn.Linear(attn_dim_size, g_dim_size).to(device)
        self.fc = nn.Linear(2*rnn_dim_size, num_class)
        # GRU network
        self.gru = nn.GRU(attn_dim_size, rnn_dim_size, 2, batch_first=True, dropout=drop_prob)
        # embedding for leaf codes and internal codes
        self.embed_init = nn.Embedding(leaf_codes_num + internal_codes_num, emb_dim_size).to(device)

        # DAG attention with 2D and mask
        self.dag_attention = DAGAttention2D(2 * emb_dim_size, attn_dim_size).to(device)

        # DAG attention with 2D and mask
        self.dag_attention_ans = DAGAttention2D(2 * emb_dim_size, attn_dim_size).to(device)


        # # DAG attention
        # self.dag_attention = DAGAttention(2*emb_dim_size, attn_dim_size).to(device)
        # # Calculate the DAG Attention for leaf nodes
        # attn_embed_list = []
        # for ls, ans in zip(leaves_list, ancestors_list):
        #     ls = np.array(ls).astype(np.long)
        #     ans = np.array(ans).astype(np.long)
        #
        #     leaves_emb = self.embed_init(torch.from_numpy(ls).to(device))
        #     ancestors_emb = self.embed_init(torch.from_numpy(ans).to(device))
        #
        #     attn_embd = self.dag_attention(leaves_emb, ancestors_emb)
        #     attn_embed_list.append(attn_embd)
        #
        # dag_emb = torch.cat(attn_embed_list, dim=-1)
        # self.dag_emb = dag_emb.reshape((int(dag_emb.size()[0] / self.attn_dim_size), self.attn_dim_size))
        #
        # # Calculate the DAG Attention for internal nodes
        # attn_embed_map = {}
        # for ls, ans in zip(internal_list, internal_ancestors_list):
        #     ls = np.array(ls).astype(np.long)
        #     ans = np.array(ans).astype(np.long)
        #
        #     leaves_emb = self.embed_init(torch.from_numpy(ls).to(device))
        #     ancestors_emb = self.embed_init(torch.from_numpy(ans).to(device))
        #
        #     attn_embd = self.dag_attention(leaves_emb, ancestors_emb)
        #     attn_embed_map[ans[0]] = attn_embd
        # sorted_map = dict(sorted(attn_embed_map.items()))
        # # print(sorted_map)
        # dag_emb = torch.cat(list(sorted_map.values()), dim=-1)
        # self.dag_emb_A = dag_emb.reshape((int(dag_emb.size()[0] / self.attn_dim_size), self.attn_dim_size))



    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(2, batch_size, self.rnn_dim_size).zero_().to(self.device)
        return hidden

    def forward(self, inputs_x, inputs_f, mask_seqs, lengths):

        leaves_emb = self.embed_init(self.leaves_list)
        ancestors_emb = self.embed_init(self.ancestors_list)
        dag_emb = self.dag_attention(leaves_emb, ancestors_emb, self.masks_list)

        leaves_emb = self.embed_init(self.internal_list)
        ancestors_emb = self.embed_init(self.internal_ancestors_list)
        dag_emb_A = self.dag_attention_ans(leaves_emb, ancestors_emb, self.masks_ancestors_list)

        # a = W_k A + B_k
        a = self.linear(dag_emb_A)
        # padding the last one
        padding = torch.zeros([1, self.g_dim_size], dtype=torch.float32).to(self.device)
        a = torch.cat([padding, a], dim=0)
        # embedding for internal codes and padding
        # self.embed_a = nn.Embedding(internal_codes_num + 1, g_dim_size, _weight=a).to(device)
        embed_a = nn.Embedding.from_pretrained(a, freeze=False).to(self.device)

        x = torch.tanh(torch.matmul(inputs_x, dag_emb))
        h = self.init_hidden(mask_seqs.shape[0]).data
        out, h = self.gru(x.to(self.device).float(), h)

        l = embed_a(inputs_f)

        out_re = out.unsqueeze(2)
        hl = (out_re * l).sum(dim=-1)
        mask_ans = (inputs_f > 0)
        mask_rank = (1 - mask_ans.double()) * VERY_NEGATIVE_NUMBER
        hl += mask_rank
        weights = torch.softmax(hl, dim=-1)
        weights = weights.unsqueeze(3)
        k = (weights * l).sum(dim=2)

        s = torch.cat([out, k], dim=-1)
        output = self.fc(s)
        output = torch.sigmoid(output)
        # out = torch.softmax(out, -1)
        # mask_t = torch.from_numpy(mask_seqs).to(self.device)
        mask_t = mask_seqs.unsqueeze(2)
        output = output * mask_t

        # get last valid output
        batch_size, len_seqs, dim_size = output.shape[0], output.shape[1], output.shape[2]
        out_reshape = output.view(-1, output.shape[-1])
        index_last_valid_output = [(i * len_seqs + lengths[i] - 1).long() for i in range(batch_size)]
        index_last_valid_output = torch.tensor(index_last_valid_output)
        last_valid_output = out_reshape[index_last_valid_output, :]

        return output, last_valid_output





