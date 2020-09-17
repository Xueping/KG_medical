from .modeling import BertEncoder, AttentionPooling, BertLayerNorm, PositionEmbeddings, BertEncoderDag
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from GRAM.gram_module import DAGAttention
import numpy as np
import torch

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


class DiagnosisPrediction(nn.Module):
    def __init__(self, config):
        super(DiagnosisPrediction, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_ccs_classes

        self.encoder_visit = BertEncoderDag(config)
        self.encoder_patient = BertEncoder(config)

        self.visit_pooling = AttentionPooling(config)
        self.patient_pooling = AttentionPooling(config)

        self.position_embedding = PositionEmbeddings(config)

        # self.text_cnn = TextCNN(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # embedding for leaf codes and internal codes
        # print('Add dag!----------------------')
        self.embed_init = nn.Embedding(config.num_tree_nodes, config.hidden_size)
        # Calculate the DAG Attention for leaf nodes
        self.dag_attention = DAGAttention(2 * config.hidden_size, config.hidden_size)
        attn_embed_list = []
        for ls, ans in zip(config.leaves_list, config.ancestors_list):
            ls = np.array(ls).astype(np.long)
            ans = np.array(ans).astype(np.long)
            leaves_emb = self.embed_init(torch.from_numpy(ls))
            ancestors_emb = self.embed_init(torch.from_numpy(ans))
            attn_embd = self.dag_attention(leaves_emb, ancestors_emb)
            attn_embed_list.append(attn_embd)
        dag_emb = torch.cat(attn_embed_list, dim=-1)
        leaves_dag = dag_emb.reshape((int(dag_emb.size()[0] / self.hidden_size), self.hidden_size))
        padding = torch.zeros([1, self.hidden_size], dtype=torch.float32)
        dict_matrix = torch.cat([padding, leaves_dag], dim=0)
        self.embed_dag = nn.Embedding.from_pretrained(dict_matrix, freeze=False)

        self.embed_inputs = nn.Embedding(config.code_size, self.hidden_size)

        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                visit_mask=None,
                code_mask=None,
                labels=None):

        input_tensor = self.embed_inputs(input_ids)  # bs, visit_len, code_len, embedding_dim
        input_shape = input_tensor.shape
        inputs = input_tensor.view(-1, input_shape[2], input_shape[3])  # bs * visit_len, code_len, embedding_dim

        input_tensor_dag = self.embed_dag(input_ids)
        inputs_dag = input_tensor_dag.view(-1, input_shape[2], input_shape[3])  # bs * visit_len, code_len, embedding_dim

        inputs_mask = code_mask.view(-1, input_shape[2])  # bs * visit_len, code_len

        extended_attention_mask = inputs_mask.unsqueeze(1).unsqueeze(2)  # bs * visit_len,1,1 code_len
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER
        visit_outputs = self.encoder_visit(inputs, extended_attention_mask,
                                           inputs_dag, output_all_encoded_layers=False)

        attention_mask = inputs_mask.unsqueeze(2)  # bs * visit_len,code_len,1
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * VERY_NEGATIVE_NUMBER
        visit_pooling = self.visit_pooling(visit_outputs[-1], attention_mask)
        visit_outs = visit_pooling.view(-1, input_shape[1], input_shape[3])  # bs, visit_len, embedding_dim

        # add position embedding
        visit_outs = self.position_embedding(visit_outs)

        extended_attention_mask = visit_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER
        patient_outputs = self.encoder_patient(visit_outs, extended_attention_mask, output_all_encoded_layers=False)

        attention_mask = visit_mask.unsqueeze(2)  # bs * visit_len,code_len,1
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * VERY_NEGATIVE_NUMBER
        patient_pooling = self.patient_pooling(patient_outputs[-1], attention_mask)

        prediction_scores = self.classifier(patient_pooling)
        prediction_scores = torch.sigmoid(prediction_scores)

        if labels is not None:
            # loss_fct = MultiLabelSoftMarginLoss()
            # loss = loss_fct(prediction_scores.view(-1, self.num_labels), labels)
            logEps = 1e-8
            cross_entropy = -(labels * torch.log(prediction_scores.view(-1, self.num_labels) + logEps) +
                              (1. - labels) * torch.log(1. - prediction_scores.view(-1, self.num_labels) + logEps))
            loglikelihood = cross_entropy.sum(axis=1)
            loss = torch.mean(loglikelihood)

            return loss
        else:
            return prediction_scores

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class KemceBinaryPrediction(nn.Module):
    def __init__(self, config):
        super(KemceBinaryPrediction, self).__init__()
        self.num_labels = self.num_labels
        self.bert = BertEncoder(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                input_ent=None,
                ent_mask=None,
                attention_mask=None,
                labels=None):

        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     input_ent,
                                     ent_mask,
                                     output_all_encoded_layers=False)

        prediction_scores = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return prediction_scores

