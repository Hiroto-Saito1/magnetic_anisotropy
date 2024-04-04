#!/bin/bash 
#PBS -q GroupD 

cd $PBS_O_WORKDIR 
source /opt/intel_2022/setvars.sh --force intel64
conda activate /home/hirotosaito/anaconda3

python bandfilling.py
rm -f submit.sh.*
