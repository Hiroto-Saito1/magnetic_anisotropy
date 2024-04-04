#!/bin/zsh

ESPRESSO_DIR_6=/home2/koretsune/codes/qe-6.6/
ESPRESSO_DIR_7=/home2/koretsune/codes/qe-7.0/
WANNIER_DIR=/home2/koretsune/codes/wannier90-3.1.0_dev/

mpirun -n 22 $ESPRESSO_DIR_7/bin/pw.x < scf.in > scf.out
mpirun -n 22 $ESPRESSO_DIR_7/bin/pw.x < nscf.in > nscf.out
mpirun -n 22 $ESPRESSO_DIR_7/bin/bands.x < band.in > band.out

#$WANNIER_DIR/wannier90.x -pp pwscf
#mpirun -n 22 $ESPRESSO_DIR_7/PP/src_v0.1.1_mod/pw2wannier90.x < pw2wan.in > pw2wan.out
rm -rf work

#mv pwscf_ibz.mmn pwscf.mmni
#mv pwscf_ibz.amn pwscf.amni
#mv pwscf_ibz.eig pwscf.eigi

#source /home/koretsune/anaconda3/bin/activate

#export PYTHONPATH=..
#python ../symwannier/write_full_data.py pwscf
#mpirun -n 22 $WANNIER_DIR/wannier90.x pwscf
#python ../symwannier/wannierize.py -s -S pwscf > wannierize.out
