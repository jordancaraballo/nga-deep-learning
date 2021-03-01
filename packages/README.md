# Docker Container with RAPIDS

## Building Container

```bash
docker build --tag dgxrapids:1.0 .
docker login
docker tag dgxrapids:1.0 nasanccs/dgxrapids:latest
docker push nasanccs/dgxrapids
```

## Setting Up Singularity Directories for Faster Download

Under ~/.bashrc add the following variables.

``bash
export SINGULARITY_CACHEDIR="/lscratch/jacaraba/singularity_tmp"
export SINGULARITY_TMPDIR="/lscratch/jacaraba/singularity_tmp"
export NOBACKUP="/att/gpfsfs/briskfs01/ppl/jacaraba"
``

Then, source the .bashrc file.

```bash
source ~/.bashrc
```

## Pulling and Executing Container

```bash
singularity pull docker://nasanccs/dgxrapids
```

## Executing Container

```bash
singularity shell --nv -B /att/nobackup/jacaraba:/att/nobackup/jacaraba,/att/gpfsfs/atrepo01/ILAB:/att/gpfsfs/atrepo01/ILAB dgxrapids_latest.sif
source activate rapids
```
