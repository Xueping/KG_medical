import torch
import torch.nn as nn
from KEMCE.knowledge_bert import KemceForPreTraining, DescTokenizer, EntityTokenizer, SeqsTokenizer, BertConfig
import pickle
import os
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

data_path = '../../'
code2desc_file = data_path + 'outputs/kemce/KG/code2desc.pickle'
ent_embd_file = data_path + 'outputs/kemce/KG/embeddings/CCS_TransR_entity.npy'
ent_vocab_file = data_path + 'outputs/kemce/KG/entity2id'
seqs_vocab_file = data_path + 'outputs/kemce/data/raw/mimic_vocab.txt'
seqs_file = data_path + 'outputs/kemce/data/raw/mimic.seqs'
ent_file = data_path + 'outputs/kemce/data/raw/mimic.entity'
config_json = data_path + 'src/KEMCE/kemce_config.json'

config = BertConfig.from_json_file(config_json)

seqs_tokenizer = SeqsTokenizer(seqs_vocab_file)
ent_tokenize = EntityTokenizer(ent_vocab_file)
desc_tokenize = DescTokenizer(code2desc_file)

seqs= pickle.load(open(seqs_file, 'rb'))
ents= pickle.load(open(ent_file, 'rb'))

sample_index = 0

visit_sample = seqs[sample_index]
ent_sample = ents[sample_index]
print(visit_sample)

seq_tokens, seq_input = seqs_tokenizer.tokenize(visit_sample)
ent_tokens, ent_input = ent_tokenize.tokenize(ent_sample)
desc_tokens, desc_input = desc_tokenize.tokenize(ent_sample)

print(seq_tokens)

masked_index = [5,7]
masked_words = []
for index in masked_index:
    masked_words.append(seq_tokens[index])
    seq_tokens[index] = '[MASK]'

# mask_word = seq_tokens[masked_index]
# seq_tokens[masked_index] = '[MASK]'

print(seq_tokens)

seq_input = seqs_tokenizer.convert_tokens_to_ids(seq_tokens)

ent_mask = []
for ent in ent_tokens:
    if ent != "[UNK]":
        ent_mask.append(1)
    else:
        ent_mask.append(0)
ent_mask[0] = 1
input_mask = [1] * len(seq_tokens)

type_mask = np.zeros(len(seq_tokens))
index = 0
for i, token in enumerate(seq_tokens):
    if token.startswith('[SEP'):
        index = i
        break
type_mask[index+1:] = 1

seq_input_tensor = torch.tensor([seq_input])
ent_input_tensor = torch.tensor([ent_input])
desc_input_tensor = torch.tensor([desc_input])
ent_mask_tensor = torch.tensor([ent_mask])
input_mask_tensor = torch.tensor([input_mask])
type_mask_tensor = torch.tensor([type_mask]).long()

ent_embd = np.load(ent_embd_file)
ent_embd = torch.tensor(ent_embd)
pad_embed = torch.zeros(1,ent_embd.shape[1])
ent_embd = torch.cat([pad_embed, ent_embd])
ent_embedding = nn.Embedding.from_pretrained(ent_embd, freeze=True)
ent_input_tensor = ent_embedding(ent_input_tensor)

model_dir = '../../outputs/kemce/models/pre_trained_np_100/'
# WEIGHTS_NAME = 'pre_trained_pytorch_model_epoch_4.bin'
WEIGHTS_NAME = 'pytorch_model.bin_56'
weights_path = os.path.join(model_dir, WEIGHTS_NAME)
state_dict = torch.load(weights_path)

model, _ = KemceForPreTraining.from_pretrained(model_dir, state_dict)
model.eval()
# Predict all tokens
with torch.no_grad():

    prediction, _ = model(seq_input_tensor, type_mask_tensor, ent_input_tensor,
                          desc_input_tensor, ent_mask_tensor, input_mask_tensor)

    for i, index in enumerate(masked_index):
        predicted_index = torch.argmax(prediction[0, index]).item()
        predicted_token = seqs_tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(masked_words[i], predicted_token)
