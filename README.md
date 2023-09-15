# Coding Lab: Deep Reinforcement Learning
Code for the implementations of the Coding Lab Deep Reinforcement Learning from the chair of Business Analytics and Intelligent Systems at TUM.
Initial files were provided by the chair.

State representation, algorithm, and hyperparameters can be set through the main.py file, which also starts training and logs results into /output/. Several helper functions are defined in codinglab_utils.py. To easily summarise output, you can use generate_summary_csv.py.

State representations: A legacy Greedy representation can be found in environment_dist_possible.py, environment_CNN_moving.py contains the Image-like state representation, environment contains the N-Greedy representation.
Algorithm: PPOtopGUN contains our final PPO algorithm with a parameter cnn to switch between CNN and non-CNN versions.
