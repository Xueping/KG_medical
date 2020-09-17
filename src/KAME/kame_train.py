from __future__ import print_function
from __future__ import division
import torch.optim as optim
import pickle
from GRAM.gram_helpers import calculate_dimSize, get_rootCode, build_tree, print2file
from KAME.kame_helpers import leaf2ancestors, load_data, train_model
from KAME.kame_module import KAME
import torch
import time
import random
import numpy as np


dir_path = '../../'
tree_file = dir_path + 'outputs/kame/data/mimic'
seq_file = dir_path + 'outputs/kame/data/mimic.seqs'
# label_file = dir_path + 'outputs/kame/data/mimic.3digitICD9.seqs'
label_file = dir_path + 'outputs/kame/data/mimic.ccsSingleLevel.seqs'
model_path = dir_path + 'outputs/kame/model/ccs_single_level/accuracy'
log_file = model_path + '/mimic.kame.log'
data_file = model_path + '/mimic.kame'

embd_dim_size = 100
attn_dim_size = 100
rnn_dim_size = 100
g_dim_size = 100
# Batch size for training (change depending on how much memory you have)
batch_size = 64
# Number of epochs to utils for
num_epochs = 100

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_leaves = calculate_dimSize(seq_file)
num_classes = calculate_dimSize(label_file)
num_ancestors = get_rootCode(tree_file+'.level2.pk') - num_leaves + 1


leaves_list = []
ancestors_list = []
for i in range(5, 0, -1):
    leaves, ancestors = build_tree(tree_file + '.level' + str(i) + '.pk')
    leaves_list.extend(leaves)
    ancestors_list.extend(ancestors)

internal_list = []
internal_ancestors_list = []
for i in range(4, 0, -1):
    leaves, ancestors = build_tree(tree_file + '.a_level' + str(i) + '.pk')
    internal_list.extend(leaves)
    internal_ancestors_list.extend(ancestors)

seqs = pickle.load(open(seq_file, 'rb'))
labels = pickle.load(open(label_file, 'rb'))

model = KAME(leaves_list, ancestors_list, internal_list, internal_ancestors_list, num_leaves, num_ancestors,
             embd_dim_size, attn_dim_size, rnn_dim_size, g_dim_size, num_classes, device)
# Send the model to GPU
model = model.to(device)
# print(model)

# Gather the parameters to be optimized/updated in this run. we will be updating all parameters.
params_to_update = model.parameters()
# # print parameters to be updated
# for name,param in model.named_parameters():
#     if param.requires_grad == True:
#         print("\t",name)

# Observe that all parameters are being optimized
optimizer = optim.Adadelta(params_to_update)

leaf2ans = leaf2ancestors(tree_file)
print2file('Loading data ... ', log_file)
train_set, valid_set, test_set = load_data(seqs, labels, leaf2ans, num_leaves)

data_dict = dict()
data_dict['utils'] = train_set
data_dict['val'] = valid_set
data_dict['test'] = test_set
print('done!!')
prepared_data_file = data_file + '_'+ time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.data'
print2file('prepared data saved to: {}'.format(prepared_data_file), log_file)
pickle.dump(data_dict, open(prepared_data_file, 'wb'), -1)

# Train and evaluate
model_ft, hist = train_model(model, data_dict, optimizer, device, batch_size, num_epochs,
                             num_leaves, num_classes, log_file, model_path)

pickle.dump(hist, open(model_path+'/mimic.kame.val_loss_history', 'wb'), -1)
