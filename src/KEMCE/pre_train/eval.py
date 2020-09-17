import torch
from torch import nn

from KEMCE.knowledge_bert import SeqsTokenizer, EntityTokenizer, DescTokenizer, \
    KemceForPreTraining
import numpy as np

data_path = '../../../'
dict_file = data_path + 'outputs/kemce/data/raw/mimic_pre_train_vocab.txt'
code2desc_file = data_path + 'outputs/kemce/KG/code2desc.pickle'
ent_vocab_file = data_path + 'outputs/kemce/KG/entity2id'
ent_embd_file = data_path + 'outputs/kemce/KG/embeddings/CCS_TransR_entity.npy'

ent_embd = np.load(ent_embd_file)
ent_embd = torch.tensor(ent_embd)
# padding for special word "unknown"
pad_embed = torch.zeros(1,ent_embd.shape[1])
ent_embd = torch.cat([pad_embed, ent_embd])
ent_embedding = nn.Embedding.from_pretrained(ent_embd, freeze=True)

seq_tokenizer = SeqsTokenizer(dict_file)
ent_tokenize = EntityTokenizer(ent_vocab_file)
desc_tokenize = DescTokenizer(code2desc_file)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

visit = '[CLS] D_41401 D_V4581 D_53081 D_496 D_30000 [SEP] D_4241 D_2720 D_V4581 D_4538 [SEP]'
mask_input = '[CLS] D_41401 [MASK] D_53081 D_496 D_30000 [SEP] D_4241 D_2720 D_V4581 D_4538 [SEP]'
mask_label = '[PAD] [PAD] D_V4581 [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'
ent_input_str = '[UNK] D_41401 D_V4581 D_53081 D_496 D_30000 [UNK] D_4241 D_2720 D_V4581 D_4538 [UNK]'
token_type = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
ent_mask =   [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

_, mask_input = seq_tokenizer.tokenize(mask_input)
_, mask_label = seq_tokenizer.tokenize(mask_label)
_, ent_input = ent_tokenize.tokenize(ent_input_str)
_, desc_input = desc_tokenize.tokenize(ent_input_str)

# embedding entity ids
ent_input_tensor = torch.tensor(ent_input).long()
ent_input_embed = ent_embedding(ent_input_tensor).to(device)

seq_input_tensor = torch.tensor(mask_input).long().to(device)
token_type_tensor = torch.tensor(token_type).long().to(device)
desc_input_tensor = torch.tensor(desc_input).long().to(device)
ent_mask_tensor = torch.tensor(ent_mask).to(device)
input_mask_tensor = torch.tensor(input_mask).to(device)

masked_index = 2
mask_word = 'D_V4581'

# training from pre-trained model
model, _ = KemceForPreTraining.from_pretrained('kemce_pre_trained')
model = model.to(device)
model.eval()


# Predict all tokens
with torch.no_grad():

    prediction, _ = model(seq_input_tensor, token_type_tensor, ent_input_embed,
                     desc_input_tensor, ent_mask_tensor, input_mask_tensor)
    predicted_index = torch.argmax(prediction[0, masked_index]).item()

    predicted_token = seq_tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(mask_word, predicted_token)


