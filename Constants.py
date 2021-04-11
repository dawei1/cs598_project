# Let's put all the global constants here.

# #Dataloader constants
# The dataset location on Da's local machine
DWDatasetRootDir = "/data/CheXpert"
CSVpath = DWDatasetRootDir+"/.csv"
ImagePath = DWDatasetRootDir+"/"
# The size of the subset used for local development
sizeOfsubset = 1024
# The index of the images used for the subset
# This is a list exported by pickle.
SubsetIdxPath = "./SubsetIdx.pickle"
# This is a pickled list of dictionaries.
# Each dictionary contains keys of
# 1. imagePath, the relative path to the image.
# 2. label, a multi-hot numpy array
ParsedDatasetPath = "./Dataset.pickle"

