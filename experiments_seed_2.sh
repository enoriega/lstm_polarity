#!/bin/bash

source activate 2019_polarity

for seed in {2..2}
do
	echo ${seed}

	python training.py ${seed} 

done
