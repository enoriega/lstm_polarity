#!/bin/bash

for seed in {1..5}
do
	echo ${seed}

	# python training.py ${seed} 0 0
	
	# python training.py ${seed} 0 1

	python training.py ${seed} 1 0
	
	python training.py ${seed} 1 1

	python training.py ${seed} 2 0

	python training.py ${seed} 2 1

	python training.py ${seed} 2 2


done
