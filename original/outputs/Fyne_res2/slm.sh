#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -t 1-5
#$ -N Fseassp150nc_tdep3              
#$ -wd '/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm' 
#$ -o ./outputs/stdouterr
#$ -e ./outputs/stdouterr                 
#$ -l h_rt=240:00:00 
#$ -l h_vmem=8G


# Initialise the environment modules
. /etc/profile.d/modules.sh
 
# Load Python
module load anaconda/4.3.1
source activate mypython3_6
 
# Run the program
python ./model/slm.py Fyne_vars '/Fyne_res2/Fseassp150nc_tdep3/' $SGE_TASK_ID
source deactivate mypython3_6
