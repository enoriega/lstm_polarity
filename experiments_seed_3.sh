#!/bin/bash

source activate 2019_polarity

for seed in {3..3}
do
	echo ${seed}

	python training.py ${seed}


done
