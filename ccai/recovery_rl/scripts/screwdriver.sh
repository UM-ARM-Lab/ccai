#!/bin/bash

# Recovery RL (model-based recovery)
for i in {1..10}
do
	echo "RRL MB Run $i"
	python -m rrl_main --cuda --env-name screwdriver --tau 0.0002 --replay_size 100000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.99 --eps_safe 0.25 --pos_fraction 0.3 --num_unsafe_transitions 20000 --logdir screwdriver --logdir_suffix RRL_MB --num_eps 4000 --seed $i --hidden_size 256
done
