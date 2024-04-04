#!/bin/bash 
#PBS -l nodes=1:ppn=8 
#PBS -q GroupE 

source /opt/intel_2022/setvars.sh --force intel64 
conda activate mae 
cd $PBS_O_WORKDIR 
numactl --interleave=all mpirun -n 8 python /home/hirotosaito/MaterialsProject/mag20230425/magnetic_anisotropy/src/parallel_eigval.py
rm -f submit.sh.*