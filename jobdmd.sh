#!/bin/bash
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02-00:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --export=all
#SBATCH --output=outs/%a.out # where STDOUT goes
#SBATCH --error=outs/%a.err # where STDERR goes
#SBATCH --array=0-18
module load cuda

M=$((SLURM_ARRAY_TASK_ID%6))
i=$((7+SLURM_ARRAY_TASK_ID/6))
N=10000
echo $SLURM_ARRAY_TASK_ID $M $i $N
if [ $M -eq 0 ]; then
	./dmd.py --M 1 --D 0 --filesuffix $M --filebase data/dmd${i}/ &
else
	./dmd.py --M $M --D $((N/2*M)) --filesuffix $M --filebase data/dmd${i}/ & 
fi
wait
