# Let's put all the global constants here.

# #Dataloader constants
# The dataset location on Da's local machine
DWDatasetRootDir = "/data/CheXpert-v1.0-small"
DatasetRootDir = DWDatasetRootDir
TrainCSVpath = DatasetRootDir+"/train.csv"
ValidCSVpath = DatasetRootDir+"/valid.csv"
# The size of the subset used for local development
sizeOfsubset = 8192

# This is a pickled list of dictionaries.
# Each dictionary contains keys of
# 1. 'imagePath', the relative path to the image.
# 2. 'label', a multi-hot numpy array
ParsedSubsetPath = "./Sub_Dataset.pickle"

# This is a pickled list of dictionaries.
# Each dictionary contains keys of
# 1. 'imagePath', the relative path to the image.
# 2. 'label', a multi-hot numpy array
ParsedDatasetPath = "./Full_Dataset.pickle"

# This is a pickled list of dictionaries.
# Each dictionary contains keys of
# 1. 'imagePath', the relative path to the image.
# 2. 'label', a multi-hot numpy array
ParsedValidsetPath = "./Validation_Dataset.pickle"

# Enable/disable image augmentation
ImageAugment = True

# batch size
batch_size = 256

# Random seed, set to None for random
seed = 1231245

eval_set_ratio = 0.2
