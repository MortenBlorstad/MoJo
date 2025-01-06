# Lux-AI-S3
This repo contains the Agent(s) for the Kaggle competition [*Lux AI Challenge Season 3*](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/kits/README.md).

The full game rules/specs can be found [here](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md).

## How to use

*TODO*
- [ ] Make a description of how to train agents 
- [ ] Make a description of how to run agents
- [ ] Make a description of how to submit agents 


## Team MoJo: 
**Mo**rten Blørstad & **Jø**rgen Mjaaseth

## Plan/Idea

### Components
- **Unit agent**: an agent that controls a unit. Performs tasks/missions.  
- **Mission control**: an agent assigning tasks/missions to unit agents (e.g. explore or gather points (relic))
-  **Env Model**: Model of the world/env/state/transition used to predict the next $n$ states.

*TODO*
- [ ] Make Env Model, $O_{t+1} \sim P(\cdot | O_t)$
- [ ] Make simple Mission control with two missions to explore or relic.
- [ ] Make simple Unit agents that can perform two missions explore or relic.
- [ ] Task 1
  - [ ] Sub-task 1.1
  - [ ] explore
  - [ ] relic

### State
The world is a 24x24 grid.


$O_t$: $24x24x8$

8 channels: 
- Player 1 Unit ($P_1$)
- Player 2 Unit ($P_2$)
- Nebula Tiles (N)
- Energy Nodes (E)
- Energy Void (V)
- Asteroid Tiles (A)
- Relic Nodes (R)
- Observered / in vision (O)

A state $S_t$ is $O_{t-3}:O_{t+3}$: $24x24x8x7$.
$O_{t+1}:O_{t+3}$ is given by the Env Model, $\sim P(\cdot | O_t)$ 

![state](https://github.com/user-attachments/assets/9c09c31d-b274-43fc-be4d-1934c46f2e35)

*TODO*
- [ ] Make a function that creates observation: $O_t$: $24x24x8$

### Reward 
Different rewards depending on the  type of agent and mission:
- Mission control: overall reward (e.g. our team score or our team score/opponents team score)
- Unit agent: a specific reward function for each task. 

*TODO*
- [ ] define a reward function for Mission control.
- [ ] define a reward function for Unit agent - relic.
- [ ] define a reward function for Unit agent - explore.


