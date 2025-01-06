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

*Components*
- **Unit agent**: an agent that controls a unit. Performs tasks/missions.  
- **Mission control**: an agent assigning tasks/missions to unit agents (e.g. explore or gather points (relic))
-  **Env Model**: Model of the world/env/state/transition used to predict the next $n$ states.
-  **Reward**: Different rewards depending on the  type of agent and mission:
    - Mission control: overall reward (e.g. our team score or our team score/opponents team score)
    - Unit agent: a specific reward function for each task. 




