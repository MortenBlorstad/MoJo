# Agent

This is the agent module. It contains the implementation of the agent that will be trained to play the game.

The agent consists of a mission control and workers. The mission control assigns tasks to the workers in the latent space. The workers then use the task together with the current state of the game to decide what action to take.

Observation $o$: a 26 x 24 x 24 tensor received from `univers`.
State $s$: a 1024 x 1 embedding of the observation.
Goal $g$: a 1024 x 1 tensor given by the mission control.
Action $a$: a 6 x 1 one-hot encoding representing the action to take.


The neural network architecture for the different components are located in `networks` folder. The `networks` folder contains the following files:
- `actor_critic.py`: Contains the actor critic network.
- `encoder.py`: Contains the encoder network.
- `decoder.py`: Contains the decoder network.
- `world_model.py`: Contains the world/HRSSM model network.

TODO: Implement the mission control. Need a HRSSM for world model and the Director algorithm to learn tasks.

TODO: Implement the workers. Will use PPO.


## worker
Worker Policy: $\texttt{wkr}(a_t | s_t g)$

Worker is a PPO agent that takes in the current state of the game and the goal and outputs an action.



