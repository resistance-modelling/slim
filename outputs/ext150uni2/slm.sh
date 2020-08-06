#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -t 1-5
#$ -N resfunc_ext150uni2             
#$ -wd '/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm' 
#$ -w n
#$ -o ./outputs/stdouterr
#$ -e ./outputs/stdouterr                 
#$ -l h_rt=96:00:00 
#$ -l h_vmem=32G


# Initialise the environment modules
. /etc/profile.d/modules.sh
 
# Load Python
module load anaconda/4.3.1
source activate mypython3_6
 
# Run the program
python ./model/slm_resfunc.py Fyne_vars '/res_func4/ext150uni2/' $SGE_TASK_ID
source deactivate mypython3_6
