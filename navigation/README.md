# Navigation

In this project, an agent is trained to navigate (and collect bananas!) 
in a large, square world.

A reward of +1 is provided for collecting a yellow banana, 
and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of your agent is to collect as many yellow bananas as possible 
while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, 
along with ray-based perception of objects around agent's forward direction. 
Given this information, the agent has to learn how to best select actions. 
Four discrete actions are available, corresponding to:

0 - move forward.  
1 - move backward.  
2 - turn left.  
3 - turn right.  

The task is episodic, and in order to solve the environment, 
the agent must get an average score of +13 over 100 consecutive episodes.

## Downloading the environment

Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)  
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)  
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)  
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)  

## Training the agent

Several agent classes are defined in `dqn_agent.py`, 
the models used are defined in `model.py`.
The function for training an agent is defined in `train.py`.

The jupyter notebook `Report.py` gives examples of training runs.
Training is done on a CPU.