# Let's put all the global constants here.

# #Dataloader constants
# The dataset location on Da's local machine
daDatasetRootDir = "/data/CheXpert-v1.0-small"
WeidiDatasetRootDir = "./data/CheXpert-v1.0-small"
jackDatasetRootDir = None
ryanDatasetRootDir = None
DatasetRootDir = "/home/CheXpert-v1.0-small"
TrainCSVpath = DatasetRootDir+"/train.csv"
ValidCSVpath = DatasetRootDir+"/valid.csv"
# The size of the subset used for local development
sizeOfsubset = 16384

# This is a pickled list of dictionaries.
# Each dictionary contains keys of
# 1. 'imagePath', the relative path to the image.
# 2. 'label', a multi-hot numpy array
ParsedSubsetPath = "./Sub_Dataset.pickle"
ParsedSubsetPath_wide = "./Sub_Dataset_wide.pickle"

# This is a pickled list of dictionaries.
# Each dictionary contains keys of
# 1. 'imagePath', the relative path to the image.
# 2. 'label', a multi-hot numpy array
ParsedDatasetPath = "./Full_Dataset.pickle"
ParsedDatasetPath_wide = "./Full_Dataset_wide.pickle"

# This is a pickled list of dictionaries.
# Each dictionary contains keys of
# 1. 'imagePath', the relative path to the image.
# 2. 'label', a multi-hot numpy array
ParsedValidsetPath = "./Validation_Dataset.pickle"
ParsedValidsetPath_wide = "./Validation_Dataset_wide.pickle"

# Enable/disable image augmentation
ImageAugment = True

# batch size
batch_size = 64

# Random seed, set to None for random
seed = 1231245

num_classes = 14

eval_set_ratio = 0.2

image_resize_size = 512
image_crop_size = 448

num_of_workers = 8
