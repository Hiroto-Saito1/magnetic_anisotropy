#!/bin/sh
#PBS -l nodes=1:ppn=12
#PBS -q GroupA

cd $PBS_O_WORKDIR

ESPRESSO_DIR_7=/home/koretsune/codes/qe-7.0/

mpirun -n 12 $ESPRESSO_DIR_7/bin/pw.x < scf.in > scf.out
mpirun -n 12 $ESPRESSO_DIR_7/bin/pw.x < nscf.in > nscf.out
mpirun -n 12 $ESPRESSO_DIR_7/bin/bands.x < band.in > band.out

rm -rf work
rm submit.sh.*
