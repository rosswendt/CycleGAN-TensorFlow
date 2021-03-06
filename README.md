# CycleGAN-TensorFlow
This is Ross Wendt's fork of the implementation of CycleGan using TensorFlow by Van Huyz (work in progress).

Original CycleGAN paper: https://arxiv.org/abs/1703.10593

# Application to games

The purpose of my application is to create additional stylized artwork for the PC game Hearts of Iron 4. Hand colored photos compiled by a member of the video game's community are paired with the the game's original artwork. In this way, additional artwork is made for the game.

Examples of original artwork from the game can be viewed at http://imgur.com/a/UYPkR.

![Leader 1](http://imgur.com/Ubhqd3x.png)
![Leader 2](http://imgur.com/pBw6W1g.png)

# Results

![Partially trained](http://imgur.com/GtIjquO.png)

![Fully trained](http://imgur.com/yC0pwwF.png)

Images involved in this application are historical photos of prominent individuals of World War 2, because the game is set in that period. The model was run with default parameters, and run for a duration roughly in line with the results from the original paper. Results of historical images before and after processing part way through the training process can be viewed at http://imgur.com/a/1rwpf. Note the black blotch artifact.

Results after the full training duration can be viewed at http://imgur.com/a/qFt7o. Note the prominent artifacts, but without the black blotch.

# Description of forks

uuid_runs https://github.com/rosswendt/CycleGAN-TensorFlow/tree/uuid_runs adds a uuid tag to different runs, letting tensorboard be a bit happier.

automated_inference https://github.com/rosswendt/CycleGAN-TensorFlow/tree/automated_inference is  a script for automatically running inference on a folder of files with command line support.





