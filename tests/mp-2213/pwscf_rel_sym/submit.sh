#!/bin/bash 
#PBS -q GroupC 

cd $PBS_O_WORKDIR 
source /opt/intel_2022/setvars.sh --force intel64
conda activate mae


python bandfilling.py
rm -f submit.sh.*
