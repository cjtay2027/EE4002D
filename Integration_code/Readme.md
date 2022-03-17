Code here was used in the integrated scenerio

Algos Tested:
1. DQN
2. TRPO
3. PPO
4. REINFORCE
5. PSO

Additional Comments:
- Agent decision is only invoked during the running process
- By default during maintenance mode (4,5,6,7) , the action is 0 (no maintenance) altho based on the MDP 1 or 0 would not result in the same outcome of exiting the maintenance state
- During failure (8,9), the action is 1 (maintenance) until the machine returns to state 0
- Rewards are not generated locally but taken from the mqtt broker which is updated by the physical layer after each decision it receives from the agent
- Cumulative rewards is tabulated and stored in a tensorboard logs directory which could be open for visualisation in real time
