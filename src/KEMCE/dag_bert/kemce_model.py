from .modeling import BertModel, PreTrainedBertModel, TextCNN, \
    BertOnlyMLMHead, BertOnlyNSPHead
from torch.nn import CrossEntropyLoss, MultiLabelSoftMarginLoss
import torch.nn as nn
from GRAM.gram_module import DAGAttention2D, DAGAttention
import numpy as np
import torch


class KemceDxPrediction(PreTrainedBertModel):
    def __init__(self, config):
        super(KemceDxPrediction, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_ccs_classes
        # self.num_labels = config.num_dx_classes
        self.bert = BertModel(config)
        # self.text_cnn = TextCNN(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # embedding for leaf codes and internal codes
        self.embed_init = nn.Embedding(config.num_tree_nodes, config.hidden_size)

        # # DAG attention
        # self.dag_attention = DAGAttention2D(2 * config.hidden_size, config.hidden_size)
        #
        # leaves_emb = self.embed_init(config.dag_leaves_list)
        # ancestors_emb = self.embed_init(config.dag_ancestors_list)
        # leaves_dag = self.dag_attention(leaves_emb, ancestors_emb, config.dag_mask_list)

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

        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                input_ent=None,
                ent_mask=None,
                attention_mask=None,
                labels=None):

        input_dag = self.embed_dag(input_ent)
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     input_dag,
                                     ent_mask,
                                     output_all_encoded_layers=False)

        prediction_scores = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = MultiLabelSoftMarginLoss()
            loss = loss_fct(prediction_scores.view(-1, self.num_labels), labels)
            return loss
        else:
            return prediction_scores


class KemceFTPrediction(PreTrainedBertModel):
    def __init__(self, config):
        super(KemceFTPrediction, self).__init__(config)
        self.num_labels = self.num_labels
        self.bert = BertModel(config)
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


class KemceForPreTraining(PreTrainedBertModel):
    def __init__(self, config):
        super(KemceForPreTraining, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.cls_seq = BertOnlyNSPHead(config)

        # embedding for leaf codes and internal codes
        self.embed_init = nn.Embedding(config.num_tree_nodes, config.hidden_size)

        # # DAG attention
        # self.dag_attention = DAGAttention2D(2 * config.hidden_size, config.hidden_size)
        #
        # leaves_emb = self.embed_init(config.dag_leaves_list)
        # ancestors_emb = self.embed_init(config.dag_ancestors_list)
        # leaves_dag = self.dag_attention(leaves_emb, ancestors_emb, config.dag_mask_list)

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

        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                input_dag=None,
                mask_dag=None,
                mask_attention=None,
                masked_lm_labels=None,
                next_sent_labels=None):

        # the shape is (batch, word_len, hidden_size)
        input_dag = self.embed_dag(input_dag)

        sequence_output, pooled_output = self.bert(input_ids,
                                                   token_type_ids,
                                                   mask_attention,
                                                   input_dag,
                                                   mask_dag,
                                                   output_all_encoded_layers=False)

        prediction_scores = self.cls(sequence_output)
        seq_relationship_score = self.cls_seq(pooled_output)

        if masked_lm_labels is not None and next_sent_labels is not None:
            loss_fct = CrossEntropyLoss()

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.code_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sent_labels.view(-1))
            total_lost = masked_lm_loss + next_sentence_loss
            # total_lost = masked_lm_loss
            return total_lost
        else:
            return prediction_scores, seq_relationship_score
