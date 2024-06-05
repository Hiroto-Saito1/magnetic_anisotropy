#PBS -l nodes=1:ppn=10 
#PBS -q GroupE 

source /opt/intel_2022/setvars.sh --force intel64 
conda activate mae 
cd $PBS_O_WORKDIR 
numactl --interleave=all mpirun -n 10 python /home/hirotosaito/magnetic_anisotropy/src/angle_dep_light.py
rm -f submit.sh.*