import torch
import torch.nn as nn
from KEMCE.knowledge_bert import DescTokenizer, EntityTokenizer, SeqsTokenizer, BertConfig, KemceForPreTraining
from KEMCE.knowledge_bert.optimization import BertAdam
import os
import pickle
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

visit_sample = seqs[0]
ent_sample = ents[0]

seq_tokens, seq_input = seqs_tokenizer.tokenize(visit_sample)
ent_tokens, ent_input = ent_tokenize.tokenize(ent_sample)
desc_tokens, desc_input = desc_tokenize.tokenize(ent_sample)

masked_index = 7
seq_tokens[masked_index] = '[MASK]'
mask_labels = [0] * len(seq_tokens)
mask_labels[masked_index] = seq_input[masked_index]
print(visit_sample)
print(seq_input)
print(mask_labels)

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
    if token.startswith('[SEP]'):
        index = i
        break
type_mask[index+1:] = 1

seq_input_tensor = torch.tensor([seq_input])
ent_input_tensor = torch.tensor([ent_input])
desc_input_tensor = torch.tensor([desc_input])
ent_mask_tensor = torch.tensor([ent_mask])
input_mask_tensor = torch.tensor([input_mask])
type_mask_tensor = torch.tensor([type_mask]).long()
mask_labels = torch.tensor([mask_labels])
next_sent_label = torch.tensor([[0]])

ent_embd = np.load(ent_embd_file)
ent_embd = torch.tensor(ent_embd)
pad_embed = torch.zeros(1,ent_embd.shape[1])
ent_embd = torch.cat([pad_embed, ent_embd])
ent_embedding = nn.Embedding.from_pretrained(ent_embd, freeze=True)
ent_input_tensor = ent_embedding(ent_input_tensor)


model = KemceForPreTraining(config)
model.train()

for _ in range(200):

    loss = model(seq_input_tensor, type_mask_tensor, ent_input_tensor, desc_input_tensor, ent_mask_tensor,
                 input_mask_tensor, mask_labels, next_sent_label)

    # Prepare optimizer
    params_to_update = model.parameters()
    # optimizer = optim.Adadelta(params_to_update)
    optimizer = BertAdam(params_to_update, lr=5e-5)

    loss.backward()
    # loss.backward()
    optimizer.step()
    print(loss.item())

# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(data_path, 'outputs/kemce/models/',  "pytorch_model.bin")
torch.save(model_to_save.state_dict(), output_model_file)
