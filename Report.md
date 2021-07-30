# Project 1: Banana Collector

This project train an agent to navigate and collect bananas in a large, square world.



### Learning Algorithms:

I used Deep Q-Neural Network (DQN) underlying learning algorithms for training the agent. Additionally I have used replay buffer to improve the training. Experience Replay is that save the experience(states, actions, rewards and next states in episodes) to Replay buffer. Therefore get a random samples to learn from saved experience. So we can learn the agent from much more experiences and there less bias in the training.


### Chosen Hyperparameters:

Below are the hyper parameters which are used in this project:

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 
UPDATE_FREQUENCY = 4    # how often to update the network


### Architecture of Neural Network Used:



### Plot of Rewards:



### Ideas for Future Work:




