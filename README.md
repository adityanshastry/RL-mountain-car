# RL-mountain-car
Implementation of RL methods of SARSA and Q-Learning to solve mountain car problem.

Below mentioned are the compilation instructions to run the Mountain Car domain on a system

Prerequisites:
1) Python 2.7x or higher
2) numpy - for numerical operations
3) sklearn - for computing the fourier basis constant vectors
3) joblib - to parallelize the runs
4) matplotlib - to plot the graphs

Instructions:
1) Place the code folder in any desired location
2) Q_2.py finds the best parameters. It accepts 2 parameters - function to run, and number of processes respectively.
   Parameter 1 can take 3 values : 1 - sarsa, 2 - Q-Learning, 3 - Plot graphs </br>
   Parameter 2 takes integer values. Ideal value for a laptop would be 10.

   $> python Q_2.py 1 10 -- runs sarsa function with 10 parallel processes

   For convenience, the results from the best parameters are present in the results folder. </br>
   So, graphs can be plotted directly using the below instruction
   
   $> python Q_2.py 3


3) Q_3.py runs the code for given parameters. It accepts the same parameters as Q_2, and has similar configurations.
   Again, for convenience, the from a run are included in the results folder. </br>
   So, graphs can be plotted directly using the below instruction
   
   $> python Q_3.py 3


As a side note, the environment has been designed to mimic the open AI environment for Mountain-Car.
So, openAI environment could be instantiated, and used to check the performance.
