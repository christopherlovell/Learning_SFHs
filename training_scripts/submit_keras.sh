#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -o output/
#$ -l cuda.devices=1

module purge
. /etc/profile.d/modules.sh
module load easybuild/software
module load TensorFlow/1.8.0-fosscuda-2018a-Python-3.6.4

source ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/research/astro/highz/Students/Chris/spectacle

echo $PYTHONPATH

# get array of files
# shopt -s nullglob
# arr=(train_*)
# file=${arr[$SGE_TASK_ID]}
# echo $file
## parallel HDF5 not installed, so must stagger execution to prevent locking
# echo $SGE_TASK_ID
# let "time = $SGE_TASK_ID * 10"
# echo $time
# sleep "$time"
# python $file

# 10 files total
# python train_cnn_intrinsic.py 
# python train_cnn_intrinsic_noise50_x4.py

# python train_cnn_dust.py 
# python train_cnn_dust_noise50.py
# python train_cnn_dust_noise50_x4.py
# python train_cnn_dust_noise20.py
# python train_cnn_dust_noise20_x4.py

python train_cnn_eagle_intrinsic_noise50_x4.py
python train_cnn_eagle_dust.py 
python train_cnn_eagle_dust_noise50.py
python train_cnn_eagle_dust_noise50_x4.py

