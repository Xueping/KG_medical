from .modeling import BertModel, PreTrainedBertModel, TextCNN, \
    BertOnlyMLMHead, BertOnlyNSPHead
from torch.nn import CrossEntropyLoss, MultiLabelSoftMarginLoss
import torch.nn as nn


class KemceDxPrediction(PreTrainedBertModel):
    def __init__(self, config):
        super(KemceDxPrediction, self).__init__(config)
        self.num_labels = config.num_ccs_classes
        # self.num_labels = config.num_dx_classes
        self.bert = BertModel(config)
        self.text_cnn = TextCNN(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                input_ent=None,
                input_desc=None,
                ent_mask=None,
                attention_mask=None,
                labels=None):

        input_desc = self.text_cnn(input_desc)
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     input_ent,
                                     input_desc,
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
        self.text_cnn = TextCNN(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                input_ent=None,
                input_desc=None,
                ent_mask=None,
                attention_mask=None,
                labels=None):

        input_desc = self.text_cnn(input_desc)
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     input_ent,
                                     input_desc,
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
        self.bert = BertModel(config)
        self.text_cnn = TextCNN(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.cls_seq = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                input_ents=None,
                input_desc=None,
                ent_mask=None,
                attention_mask=None,
                masked_lm_labels=None,
                next_sent_labels=None):

        input_desc = self.text_cnn(input_desc)
        sequence_output, pooled_output = self.bert(input_ids,
                                                   token_type_ids,
                                                   attention_mask,
                                                   input_ents,
                                                   input_desc,
                                                   ent_mask,
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
