#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N               
#$ -cwd                  
#$ -l h_rt=12:00:00 
#$ -l h_vmem=1G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh
 
# Load Python
module load anaconda
module load python/3.4.3
 
# Run the program
