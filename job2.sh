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
#SBATCH --array=1-500
module load cuda

seed=$SLURM_ARRAY_TASK_ID
for i in `seq 1 9`; do
	N=$((10000*i))
	dK=$((2*N/10))
	KS="0 $((dK)) $((2*dK)) $((3*dK)) $((4*dK)) $((5*dK))"
	for K in $KS; do 
	echo $N $K $dK
	mkdir -p data/$N/$K
	if [ $N -lt 40000 ]; then
		./kuramoto -N $N -K $K -s $seed -c 1.75 -t 100 -d 0.01 -nvDR data/$N/$K/$seed > /dev/null
	else
		./kuramoto -N $N -K $K -s $seed -c 1.75 -t 100 -d 0.01 -nvDAR data/$N/$K/$seed > /dev/null
	fi
done
done
