#!/bin/bash
# usage: sbatch -J <name> run.sbatch <deck> <name> [key=value ...]    (epyc node, 16 OpenMP threads)
#SBATCH -p epyc
#SBATCH -c 16
#SBATCH -o slurm-%x.out
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OMP_PROC_BIND=false
deck=$1; name=$2; shift 2
QUARZ=${QUARZ:-../../build/quarz}
start=$(date +%s)
$QUARZ $deck output.dir=out_$name "$@" > log_$name.txt 2>&1
end=$(date +%s)
echo "$name wall $((end - start)) s, threads $OMP_NUM_THREADS, core-h $(echo "($end - $start) * $OMP_NUM_THREADS / 3600" | bc -l)" >> timings.txt
