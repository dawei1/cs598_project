'''
This module is used for playing with the ResNet portion of the model, for example
to ensure tha the output sizes are as expected.
'''
import random
from Dataloader_2 import *
from ResNet import *
import Constants


full_train_dataset, sub_train_dataset, val_train_dataset = get_dataset()

sub_train_dataset_len = sub_train_dataset.__len__()
eval_set_size = int(sub_train_dataset_len * Constants.eval_set_ratio)
train_set_size = sub_train_dataset_len - eval_set_size
train_dataset, eval_dataset = torch.utils.data.random_split(sub_train_dataset, [train_set_size, eval_set_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)
val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model = get_resnet_model()
for x, y in train_loader:
    out = model(x)
    print(out.shape)
    break
