#!/bin/bash

# This script will perform a grid experiment on a number of sites with
# different treatment policies and other hyperparameters. Modify the script as
# you wish.

# List of experiments to try upon
EXPERIMENTS=(Fyne_complete Linnhe_complete clique cycle bipath)

# Output directory
OUTPUT_FOLDER=./outputs

# Number of trials
NTRIALS=1000

# Number of parallel processes to spawn. Note that each simulation will be single-threaded. The greater the better, but note that it *should* (although it needs not) be a divisor of NTRIALS
NPAR=50

# For bernoulli experiments, the list of defection probabilities to use
PROBAS=(0.0 0.2 0.8 1.0)

# Available treatments
TREATMENT_TYPES=(emb thermolicer cleanerfish)

TREATMENT_FREQ=(30 60)

# 1 to disable logging, 0 to keep enabled
_BENCH_COMMAND="slim benchmark"

_OPTIONS="--trials=$NTRIALS --parallel-trials=$NPAR --quiet"

bernoulli () {
	$_BENCH_COMMAND $OUTPUT_FOLDER/$1_bernoulli $2 $1_bernoulli \
		--treatment-strategy=bernoulli --defection-proba=$3 \
		$_OPTIONS
}

mosaic () {
	$_BENCH_COMMAND $OUTPUT_FOLDER/$1_mosaic $2 $1_mosaic \
		--treatment-strategy=mosaic \
		$_OPTIONS
}


regular () {
	local treatment_freq=""
	local exp_name="$1_untreated"

	if [ -n $3 ] && [ -n $4 ] ; then
		treatment_freq="--recurrent-treatment-type=$3 --recurrent-treatment-frequency=$4"
		exp_name="$1_regular_$3_$4"
	fi

	$_BENCH_COMMAND $OUTPUT_FOLDER/$exp_name $2 $exp_name \
		--treatment-strategy=untreated $treatment_freq \
		$_OPTIONS
}

main () {
	for loch in ${EXPERIMENTS[@]}; do
		local exp_folder=config_data/$loch
		# Bernoullian experiments
		for defection_proba in ${PROBAS[@]}; do
			bernoulli $loch $exp_folder $defection_proba
		done

		# Mosaic experiments
		mosaic $loch $exp_folder

		regular $loch $exp_folder

		for tt in ${TREATMENT_TYPES[@]}; do
			for tf in ${TREATMENT_FREQ[@]}; do
				regular $loch $exp_folder $tt $tf
			done
		done
	done
}

main
