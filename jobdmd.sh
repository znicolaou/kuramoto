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
#SBATCH --output=outs/%a.out # where STDOUT goes
#SBATCH --error=outs/%a.err # where STDERR goes
#SBATCH --array=0-17
module load cuda

i0=4
M=$((SLURM_ARRAY_TASK_ID/3))
i=$((i0+SLURM_ARRAY_TASK_ID%3))
N=`head data/dmd${i}/0.out -n 1 | cut -d' ' -f1`
echo "$SLURM_ARRAY_TASK_ID data/dmd${i} M=$M  N=$N"
if [ $M -eq 0 ]; then
	./dmd.py --M 1 --D 0 --seed 100 --rank 5000 --runpseudo 1 --load 0 --filesuffix ${M} --mem 50GB --filebase data/dmd${i}/ &
else
	./dmd.py --M $M --D $((N/2*M)) --seed 100 --rank 5000 --runpseudo 1 --load 0 --filesuffix ${M} --mem 40GB --filebase data/dmd${i}/ & 
fi
wait
