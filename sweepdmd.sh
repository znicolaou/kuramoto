for d in `seq 4 5`; do
	./dmd2.py --M $((d)) --D $((1000*(d-1))) --filesuffix $d --filebase data/dmd4/ > dmd4_${d}.out 2>&1 &
	./dmd2.py --M $((d)) --D $((1000*(d-1))) --filesuffix $d --filebase data/dmd5/ > dmd5_${d}.out 2>&1 &
	./dmd2.py --M $((d)) --D $((1000*(d-1))) --filesuffix $d --filebase data/dmd6/ > dmd6_${d}.out 2>&1 &
	wait
done
