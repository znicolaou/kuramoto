#!/bin/bash
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02-00:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --export=all
#SBATCH --output=outs/%a_r.out # where STDOUT goes
#SBATCH --error=outs/%a_r.err # where STDERR goes
#SBATCH --array=0-17
module load cuda

i0=4
M=$((SLURM_ARRAY_TASK_ID%6))
i=$((i0+SLURM_ARRAY_TASK_ID/6))
N=`head data/dmd${i}/0.out -n 1 | cut -d' ' -f1`
echo "$SLURM_ARRAY_TASK_ID data/dmd${i} M=$M  N=$N"
if [ $M -eq 0 ]; then
	./dmd2.py --M 1 --D 0 --seed 100 --rank 3000 --filesuffix ${M}_r --filebase data/dmd${i}/ &
	#./dmd2.py --M 1 --D 0 --seed 100 --filesuffix ${M} --filebase data/dmd${i}/ &
else
	./dmd2.py --M $M --D $((N/2*M)) --seed 100 --rank 3000 --filesuffix ${M}_r --filebase data/dmd${i}/ & 
	#./dmd2.py --M $M --D $((N/2*M)) --seed 100 --filesuffix ${M} --filebase data/dmd${i}/ & 
fi
wait
