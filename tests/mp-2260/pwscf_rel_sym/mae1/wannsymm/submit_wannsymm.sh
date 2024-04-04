#!/bin/sh
#PBS -l nodes=1:ppn=1
#PBS -q GroupD

cd $PBS_O_WORKDIR
source /opt/intel_2022/setvars.sh intel64
numactl --interleave=all mpirun -np 1 /home/hirotosaito/WannSymm/bin/wannsymm.x wannsymm.in
rm submit_wannsymm.sh.*