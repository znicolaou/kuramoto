#!/bin/bash
#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=02-00:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --export=all
#SBATCH --output=outs/%a.out # where STDOUT goes
#SBATCH --error=outs/%a.err # where STDERR goes
#SBATCH --array=1-750
module load cuda

seed=$SLURM_ARRAY_TASK_ID
for i in `seq 1 1`; do
	N=$((5000*i))
	KS="0 2 $((5000*i))"
	for K in $KS; do 
	echo $N $K $dK
	mkdir -p data/$N/$K
	mkdir -p data/${N}_lorentz/$K
	./kuramoto -N $N -K $K -s $seed -c 1.75 -t 100 -d 0.01 -nvDR data/$N/$K/$seed > /dev/null
	./kuramoto -N $N -K $K -s $seed -c 1.75 -t 100 -d 0.01 -vDR data/${N}_lorentz/$K/$seed > /dev/null
done
done
