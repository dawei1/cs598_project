# Let's put all the global constants here.

# #Dataloader constants
# The dataset location on Da's local machine
DWDatasetRootDir = "/data/CheXpert-v1.0-small"
DatasetRootDir = DWDatasetRootDir
TrainCSVpath = DatasetRootDir+"/train.csv"
ValidCSVpath = DatasetRootDir+"/valid.csv"
ImagePath = DatasetRootDir+"/"
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

# Random seed
seed = 1231245
