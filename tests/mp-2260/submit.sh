#!/bin/bash 
#PBS -q GroupD 

source /opt/intel_2022/setvars.sh --force intel64
conda activate mae
cd $PBS_O_WORKDIR 
python /home/hirotosaito/MaterialsProject/mag20230425/MAE_programs/src/lam_dep.py  
rm -f submit.sh.*