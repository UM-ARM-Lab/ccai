#!/bin/bash


# Recovery RL (model-based recovery)
for i in {1..10}
do
	echo "RRL MB Run $i"
	python -m rrl_main --cuda --env-name navigation1 --use_recovery --gamma_safe 0.8 --eps_safe 0.3 --logdir navigation1 --logdir_suffix RRL_MB --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

