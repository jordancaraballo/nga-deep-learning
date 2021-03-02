# NGA Deep Learning

This repository includes a set of general scripts to preprocess, train, and predict
land cover and land use characteristics from very high-resolution imagery in GeoTIF format.

## Business Case

The following repository performs semantic segmentation classification over GeoTIF data using
convolutional neural networks (CNN), particularly a modified UNet.

## Table of Contents

1. [ Computational Resources ](#Computational_Resources)
   * [ Conda Environment Installation ](#Conda_Environment_Installation)
   * [ Container Environment Installation ](#Container_Environment_Installation)
2. [ How-To ](#How_To)
2. [ Authors ](#Authors)
3. [ Contributors ](#Contributors)
4. [ References ](#References)
5. [ Acknowledgments ](#Acknowledgments)

## Computational Resources <a name="Computational_Resources"></a>

A system with NVIDIA GPUs is required to run the scripts located in this repository.

### Conda Environment Installation <a name="Conda_Environment_Installation"></a>

Installing conda environment.

``` bash
module load anaconda
git clone https://github.com/jordancaraballo/nga-deep-learning.git
cd nga-deep-learning; conda env create -f requirements/environment.yml;
```

Note: Using Tensorflow 2.2 at this time since NCCS GPU Cluster does not have CUDA 11.x available as of 12/16/2020.
When CUDA 11.0 is available, then intstall cudatoolkit=10.2 and tensorflow=2.4 or greater.

### Container Environment Installation <a name="Container_Environment_Installation"></a>

The following section describes how to build, pull, and use a Docker/Singularity container to execute
the code provided in this repository.

#### Building Container

This repository includes a Docker file to build the container. The base image is taken from NVIDIA
NGC container repository with the RAPIDS environment.

```bash
git clone https://gitlab.nccs.nasa.gov/jacaraba/nga-deep-learning.git
cd nga-deep-learning/packages/
docker build --tag nga-deeplearning:latest .
docker login
docker tag nga-deeplearning:latest nasanccs/nga-deeplearning:latest
docker push nasanccs/nga-deeplearning
```

#### Downloading Container

Docker containers can be pulled as Singularity containers to be executed on HPC environments. The
following commands allow the download of the container from DockerHub.

```bash
singularity pull docker://docker.io/nasanccs/nga-deeplearning:latest
```

#### Executing Container

For Singularity containers to have access to other paths within the HPC environment, we need to bind
directories to particular locations in the container. The container provided by this project has a built-in
conda environment that includes all software dependencies to run this project. To shell into the container
and activate this environment:

```bash
singularity shell --nv -B /att/nobackup/$user:/att/nobackup/$user,/att/gpfsfs/atrepo01/ILAB:/att/gpfsfs/atrepo01/ILAB nga-deeplearning_latest.sif
source activate rapids
```

## How-To

### Expected Input Data

The following scripts expect data in GeoTIF format. There should be a file with raw data, preferebly TOA corrected, and a file with
labels or the mask to map with. The expected data file can have anywhere from 3 to N number of channels/bands, while the mask file should
have a single channel/band with integer values. Each integer value representing a class to classify. At the end of the day, the directories
should look similar to the following example:

```bash
/att/data/data01.tif, /att/data/data02.tif, /att/data/data03.tif
/att/labels/labels01.tif, /att/labels/labels02.tif, /att/labels/labels03.tif
```

Example input shape are (5000,5000,6) for data rasters, and (5000,5000) for mask files. The information regarding the data will be stored
in a CSV file located under scripts/config/data.csv, with the following format:

```bash
data,label,ntiles_train,ntiles_val,ymin,ymax,xmin,xmax
Keelin00_20120130_data.tif,cm_Keelin00_20120130_new_2.tif,500,1000,0,50,0,50
Keelin00_20180306_data.tif,cm_Keelin00_20180306_new_2.tif,500,1000,0,50,0,50
Keelin01_20151117_data.tif,cm_Keelin01_20151117_new.tif,500,1000,0,50,0,50
```

Where:

* data: is the filename with the data values
* label: is the filename with the mask values
* ntiles_train: number of training tiles to extract from the data file
* ntiles_val: number of validation tiles to extract from the data file
* ymin,ymax,xmin,xmax: values to exclude from training dataset to include in validation tiles

### Configuration File

### Preprocess

In this section of the pipeline we take raw raster and masks, and we generate training and validation
datasets to work with. The following script will save NPZ files into local storage with the train and mask
mappings taken from the data.csv file. Parameters are taken from the Configuration File. Expect to find two
new directories on the ROOT DIR specified on the Configuration file with an NPZ file per raster. These NPZ
files contain the dataset mappings with 'x' for data, and 'y' for the label.

if specified, data will be stanrdardized using local standardization.

```bash
python preprocessing.py
```

### Train

In this section of the pipeline we proceed to train the model. Please refer to the Configuration file for more
details on parameters required for training. The main script will: read the data files, map them into TensorFlow
datasets, initialize the UNet model, and proceed with training. A model (.h5 file) will be saved for each epoch
that improved model performance.

You might notice that the first epoch of the model takes a long time to train. This is expected since the model
is being optimized during the first epoch. Distributed training across multiple GPUs is enabled, together with
mixed precission for additional performance enhancements. Upcoming modifications will include A100 performance
enhancements and Horovod for multi-node training.

```bash
python train.py
```

### Predict

In this section we proceed to train the desired data taking one of the saved models, and using it for predictions.
Please refer to the Configuration file for more details on parameters required for predicting. Ideally, all images to
predict will be stored in the same directory ending on .tif extensions. The script will go through every file, predict
it and output both the predicted GeoTIF and the probabilities arrays.

```bash
python predict.py
```

## Authors

* Jordan Alexis Caraballo-Vega, <jordan.a.caraballo-vega@nasa.gov>

## Contributors

* Mary Aronne, <mary.aronne@nasa.gov>

## References

[1] Chollet, François; et all, Keras, (2015), GitHub repository, https://github.com/keras-team/keras. Accessed 13 February 2020.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, https://github.com/pytorch/pytorch. Accessed 13 February 2020.

[3] Google Brain Team; et all, TensorFlow, (2015), GitHub repository, https://github.com/tensorflow/tensorflow. Accessed 13 February 2020.

## Acknowledgments

* Mark Carroll, NASA Center for Climate Simulation, 606.2
* Mary Aronne, NASA Center for Climate Simulation, 606.2
* Dan Duffy, NASA Center for Climate Simulation, 606.2
* Laura E. Carriere, NASA Center for Climate Simulation, 606.2
