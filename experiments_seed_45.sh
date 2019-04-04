#!/bin/bash

source activate 2019_polarity

for seed in {4..5}
do
	echo ${seed}

	python training.py ${seed} 0 0
	
	python training.py ${seed} 0 1

	python training.py ${seed} 1 0

	python training.py ${seed} 1 1


done
