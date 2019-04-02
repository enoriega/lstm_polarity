#!/bin/bash


source activate 2019_polarity

for seed in {4..5}
do
	echo ${seed}

	python training.py ${seed}

done
