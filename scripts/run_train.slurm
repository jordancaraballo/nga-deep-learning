#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=unet_train_1

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4

## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file

## Slurm can send you updates via email
##SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
##SBATCH --mail-user=<GMUnetID>@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=64G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=00-08:00  # Total time needed for job: Days-Hours:Minutes


## This script will request the default 1 CPU and 2GB of memory


## Load the relevant modules needed for the job
#module load singularity   (not needed at the moment on hopper)

## Run your program or script

echo  "$(which singularity)"
singularity instance start --nv -B /home/mle35:/home/mle35,/scratch/mle35:/scratch/mle35  /home/mle35/rapidsai.sif rapids
cd /home/mle35/nga-deep-learning/scripts
singularity run instance://rapids python train.py 
