#!/bin/bash
#SBATCH --account=isaac-utk0437
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=01-00:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --export=all
#SBATCH --output=outs/dmd%a.out # where STDOUT goes
#SBATCH --error=outs/dmd%a.err # where STDERR goes
#SBATCH --array=0-9
module load cuda

M=5
N=10000
filebase0='data/dmd'
Ks=(`ls ${filebase0}/${N}`)
K=${Ks[$SLURM_ARRAY_TASK_ID]}
./dmd.py --M $M --D $((N/2*M)) --seed 100 --rank 5000 --runpseudo 0 --load 0 --filesuffix ${M} --mem 50GB --filebase ${filebase0}/${N}/${K}/ 
rm -rf ${filebase0}/${N}/${K}/${M}X0
rm -rf ${filebase0}/${N}/${K}/${M}X
rm -rf ${filebase0}/${N}/${K}/${M}v
