from __future__ import print_function
from __future__ import division
import pickle
import torch
from GRAM.gram_helpers import eval_model, calculate_dimSize, get_rootCode, build_tree


dir_path = '../../outputs/gram/'
tree_file = dir_path + 'data/mimic'
seq_file = dir_path + 'data/mimic.seqs'
# label_file = dir_path + 'outputs/mimic.3digitICD9.seqs'
label_file = dir_path + 'data/mimic.ccsSingleLevel.seqs'

log_file = dir_path + 'model/ccs_single_level/mimic.gram.log'
# gram_model_file = dir_path + 'model/ccs_single_level/mimic.gram_2020-09-11_14-40-14.model'
# gram_data = dir_path + 'model/ccs_single_level/mimic.gram.data_2020-09-11-14-29-14.data'

# gram_model_file = dir_path + 'model/ccs_single_level/mimic.gram_100_2020-09-11_19-56-34.model'
gram_model_file = dir_path + 'model/ccs_single_level/mimic.gram_epoch(99)_2020-09-11_19-56-34.model'

gram_data = dir_path + 'model/ccs_single_level/mimic.gram.data_2020-09-11-16-49-51.data'

# Batch size for training (change depending on how much memory you have)
batch_size = 100
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

num_leaves = calculate_dimSize(seq_file)
num_classes = calculate_dimSize(label_file)
num_ancestors = get_rootCode(tree_file+'.level2.pk') - num_leaves + 1


leaves_list = []
ancestors_list = []
for i in range(5, 0, -1):
    leaves, ancestors = build_tree(tree_file + '.level' + str(i) + '.pk')
    leaves_list.extend(leaves)
    ancestors_list.extend(ancestors)

torch.nn.Module.dump_patches = True
model = torch.load(gram_model_file)
model = model.to(device)
print(model)


print('Loading data ... ')
data_dict = pickle.load(open(gram_data, 'rb'))

# Evaluate
eval_model(model, data_dict, device, batch_size, num_leaves, num_classes, log_file)


