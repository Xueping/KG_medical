import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from KEMCE.dataset import prepare_data, BERTDataset, collate_mlm
from KEMCE.knowledge_bert import SeqsTokenizer, EntityTokenizer, DescTokenizer, \
    KemceForPreTraining, BertAdam, BertConfig
import numpy as np


data_path = '../../../'

out_dir = data_path + 'outputs/kemce/data/raw/'
seqs_file = data_path + 'outputs/kemce/data/raw/mimic_pre_train.seqs'
dict_file = data_path + 'outputs/kemce/data/raw/mimic_pre_train_vocab.txt'
code2desc_file = data_path + 'outputs/kemce/KG/code2desc.pickle'
ent_vocab_file = data_path + 'outputs/kemce/KG/entity2id'
ent_embd_file = data_path + 'outputs/kemce/KG/embeddings/CCS_TransR_entity.npy'
config_json = data_path + 'src/KEMCE/kemce_config.json'

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

pair_seqs, pair_seqs_mask, pair_seqs_mask_label = prepare_data(seq_tokenizer.ids_to_tokens, seqs_file, out_dir)
dataset = BERTDataset(pair_seqs, pair_seqs_mask, pair_seqs_mask_label, seq_tokenizer, ent_tokenize, desc_tokenize)

train_data_loader = DataLoader(dataset, batch_size=32,
                               collate_fn=lambda batch: collate_mlm(batch, ent_embedding),
                               num_workers=0, shuffle=True)

config = BertConfig.from_json_file(config_json)

# training from scratch
model = KemceForPreTraining(config)

# training from pre-trained model
# model, _ = KemceForPreTraining.from_pretrained('kemce_pre_trained')

print(model)
model = model.to(device)
model.train()
num_epoch = 5

# Prepare optimizer
params_to_update = model.parameters()
# optimizer = optim.Adam(params_to_update)
optimizer = BertAdam(params_to_update, lr=5e-4)

for epoch in range(num_epoch):

    dataiter = iter(train_data_loader)
    total_steps = len(dataiter)

    running_loss = 0

    for step, data in enumerate(dataiter):
        seq_input_tensor = data['mlm_input'].to(device)
        token_type_tensor = data['token_type'].to(device)
        ent_input_tensor = data['ent_input'].to(device)
        desc_input_tensor = data['desc_input'].to(device)
        ent_mask_tensor = data['ent_mask'].to(device)
        input_mask_tensor = data['input_mask'].to(device)
        mask_labels = data['mlm_label'].to(device)
        next_sent_label = data['next_sent'].to(device)

        loss = model(seq_input_tensor,
                     token_type_tensor,
                     ent_input_tensor,
                     desc_input_tensor,
                     ent_mask_tensor,
                     input_mask_tensor,
                     mask_labels,
                     next_sent_label)

        loss.backward()
        # loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()

        if step % 100 == 0:
            print('loss in step {}/{} of epoch {}: {}'.format(step, total_steps, epoch + 1, running_loss/(step+1)))

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(data_path, 'outputs/kemce/models/',
                                     "pre_trained_pytorch_model_epoch_"+str(epoch+1)+".bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    epoch_loss = running_loss / total_steps
    print('loss of epoch {}: {}'.format(epoch + 1, epoch_loss))

# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(data_path, 'outputs/kemce/models/', "pre_trained_pytorch_model.bin")
torch.save(model_to_save.state_dict(), output_model_file)
