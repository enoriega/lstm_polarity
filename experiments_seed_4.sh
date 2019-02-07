#!/bin/bash

source activate 2019_polarity

for seed in {4..4}
do
	echo ${seed}

	python training.py ${seed}


done
