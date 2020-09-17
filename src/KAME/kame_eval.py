from __future__ import print_function
from __future__ import division
import pickle
import torch
from KAME.kame_helpers import eval_model
from GRAM.gram_helpers import calculate_dimSize, get_rootCode, build_tree


dir_path = '../../outputs/kame/'
tree_file = dir_path + 'data/mimic'
seq_file = dir_path + 'data/mimic.seqs'
# label_file = dir_path + 'data/mimic.3digitICD9.seqs'
label_file = dir_path + 'data/mimic.ccsSingleLevel.seqs'

log_file = dir_path + 'model/ccs_single_level/mimic.kame.log'
# kame_model_file = dir_path + 'model/ccs_single_level/mimic.kame_2020-09-11_14-15-27.model'
# kame_data = dir_path + 'model/ccs_single_level/mimic.kame_2020-09-11-14-03-55.data'

# kame_model_file = dir_path + 'model/ccs_single_level/mimic.kame_100_2020-09-11_20-00-58.model'
kame_model_file = dir_path + 'model/ccs_single_level/mimic.kame_epoch(99)_2020-09-11_20-00-58.model'
kame_data = dir_path + 'model/ccs_single_level/mimic.kame_2020-09-11-16-49-19.data'

# Batch size for training (change depending on how much memory you have)
batch_size = 64

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

torch.nn.Module.dump_patches = True
model = torch.load(kame_model_file)
model = model.to(device)
print(model)


print('Loading data ... ')
data_dict = pickle.load(open(kame_data, 'rb'))
# Evaluate
eval_model(model, data_dict, device, batch_size, num_leaves, num_classes, log_file)


