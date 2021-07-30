# Project 1: Banana Collector (Report)

This project train an agent to navigate and collect bananas in a large, square world.



### Learning Algorithms:

I used Deep Q-Neural Network (DQN) underlying learning algorithms for training the agent. Additionally I have used Experience Replay to improve the training. Experience Replay stores the experiences(states, actions, rewards and next states in episodes) to a replay buffer, then we get random samples to learn from saved experiences. In this way ,  we can learn the agent from much more experiences and there is less bias in the training.

I have also made use of Fixed Q-Targets, which essentially means that we keep two set of network weights, one target weights from which we compute the expected reward and other  weights which are updated for every experience tuple, which gives next set of actions to take.   


### Chosen Hyperparameters:

Below are the hyper parameters which are used in this project:
 <p>
buffer_size = int(1e6)  &nbsp;&nbsp;&nbsp;&nbsp;# replay buffer size <br />
batch_size = 64         &nbsp;&nbsp;&nbsp;&nbsp;# minibatch size  <br />
gamma = 0.99            &nbsp;&nbsp;&nbsp;&nbsp;# discount factor <br />
lr = 5e-4               &nbsp;&nbsp;&nbsp;&nbsp;# learning rate <br />
update_frequency = 4    &nbsp;&nbsp;&nbsp;&nbsp;# how often to update the network <br />
</p>

### Architecture of Neural Network Used:

The DQN netowrk has four layers. 

1. First layer is the input layer which has 37 nodes , which take states as input.
2. The second and third layer are intermediate layers , each have 64 nodes.
3. The fourth layer is output layer which has 4 nodes ( same as number of actions.)



### Plot of Rewards:

<img src="bctraining.png"/>

Saved Model: [bcweights.pth](bcweights.pth)



### Ideas for Future Work:

Using Double DQN and Dueling DQN to improve training.
Using Prioritized Experience Replay to further improve the training.




