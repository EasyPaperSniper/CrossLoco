# CrossLoco: Human Motion Driven Control of Legged Robots via Guided Unsupervised Reinforcement Learning #
 CrossLoco, a guided unsupervised reinforcement learning framework that simultaneously learns robot skills and their correspondence to human motions. The key innovation is to introduce a cycle-consistency-based reward term designed to maximize the mutual information between human motions and robot states.

[[paper]](https://openreview.net/forum?id=UCfz492fM8) [[website]](https://easypapersniper.github.io/projects/crossloco/crossloco.html)



# Annocement
Although the orginal version of CrossLoco is implemented based on LeggedGym, we now move our implementation to Orbit. Currently still underconstruction but the core reward and algorithm is uploaded.



# Installation
Install Orbit follow the instruction: https://isaac-orbit.github.io/orbit/source/setup/installation.html

Clone this repo under Orbit folder, then it should be fine.


# Code structure
The environment and training configuration are included in ./envs

The main algorithm and runner files are locates in ./runners





