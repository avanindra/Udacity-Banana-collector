# Udacity-Banana-collector



### Project Details

This project train an agent to navigate (and collect bananas!) in a large, square world.

<img src="env.gif"/>

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.<br/>
1 - move backward.<br/>
2 - turn left.<br/>
3 - turn right.<br/>
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.<br/>


### Getting Started

Download the environment from one of the links below. You need only select the environment that matches your operating system:<br/>

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)<br/>
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)<br/>
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)<br/>
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)<br/>
(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.<br/>

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.<br/>




### Instructions

There are two ways , one can run the banacollector agent training:<br/>

1. Run Navigation.pynb ( with zypyter notebook, this is modified version of code supplied with the udacity project assignment. You need to spcify the environment path.)
2. Run bctraining.py with supplying environment path at commandline.

The programs generate two outputs: 
1. bcweights.pth ( the network weight for the DQN network).
2. bctraining.png ( plot of average return with number of episodes)

The programs exit when average return reaches to more than 13.0.
