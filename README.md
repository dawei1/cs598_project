# CS 598 Project
Authors: Da Wei, Jack Kovach, Ryan Baker, Weidi Ouyang

## Execution Environment
This code was executed using PyTorch Deep Learning VMs on Google Cloud. These VMs
include all required packages. You can create one of these VMs using a command like
this (assuming gcloud setup is already completed):

```
gcloud compute instances create dl-training-vm-gpu --create-disk=type=pd-ssd,size=100GB --zone=us-east1-b --project=cs598-project --image-family=pytorch-latest-cpu --image-project=deeplearning-platform-release --machine-type=n1-highmem-16 --accelerator="type=nvidia-tesla-p100,count=1" --metadata="install-nvidia-driver=True" --maintenance-policy=TERMINATE
```

Then, all necessary code files need to be transfered to the VM. This can be done
with a command like this:

```
gcloud compute scp --recurse <path-to-local-directory> dl-training-vm-gpu:~/
```

You will also need to obtain data to use for the executions. The small CheXpert data
can be downloaded like this:

```
wget "http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip"
unzip CheXpert-v1.0-small.zip -d <path-to-destination>
```

## Training the Model
To train the model, you will need to do a few things. First, update the Constants.py
file with the path to your dataset (`DatasetRootDir`) and the directory prefix (`dir_prefix`).
You can also update other constants such as the batch size and the number of workers.

Next you will need to prime the dataset. You will need to choose whether you would
like to use the regular frontal PA images, or if you would like to use the frontal+lateral
concatenated images. Lets assume you want to use the normal frontal PA images. You can prime
the dataset like this:

```
PYTHONPATH=<path-to-project-root> python frontal/Dataloader_utils.py
```

Next you need to determine if you want to use the full dataset or just a subset. In
full\_model\_evaluation.py you will need to change which lines are commented out in two
different places. You need to make sure you are importing the right dataloader depending
on which type of images you chose to use (e.g. the frontal PA images). You also need to
make sure that you are using the desired full or sub dataloader. The places where these
lines may need to be changed are marked in the code.

Finally, you just need to run

```
python full_model_evaluation.py
```

Sample outputs are given in the outputs directory and there are notes at the top of each
file indicating how those particular outputs were generated.
