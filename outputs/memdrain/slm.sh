#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -t 1
#$ -N memdrain            
#$ -wd '/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm' 
#$ -w n
#$ -o ./outputs/stdouterr
#$ -e ./outputs/stdouterr                 
#$ -l h_rt=20:00:00 
#$ -l h_vmem=32G


# Initialise the environment modules
. /etc/profile.d/modules.sh
 
# Load Python
module load anaconda/4.3.1
source activate mypython3_6
 
# Run the program
python -m memory_profiler ./model/memdrain_slm.py Fyne_vars '/res_func4/memdrain/' $SGE_TASK_ID
source deactivate mypython3_6
