#!/bin/bash 
#PBS -q GroupA 

source /home/hirotosaito/anaconda3/bin/activate
cd $PBS_O_WORKDIR 
python /home/hirotosaito/MaterialsProject/mag20230526/MAE_programs/src/ma_quantities.py \ 
 input_params.toml 
rm -f submit.sh.*