# Stable Baselines 3 Simple RL Kit

This is a simple stable baselines 3 RL kit based off of part 2 of the [RL tutorial series](https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving)

`train.py` implements a very simple approach to training an RL agent to dig ice and deliver it back to factories to generate water and survive longer.

The general structure of this RL agent is that we use a heuristic policy to handle bidding and factory placement, then train the RL agent on solving the normal phase of the game. The goal of this RL agent is to survive as long as possible without worrying about growing lichen yet.

## Training

To use the training code, run `train.py --help` for help and to train an agent run

```
python train.py --n-envs 10 --log-path logs/exp_1  --seed 42
```

which trains an RL agent using the PPO algorithm with 10 parallel environments for 7,500,000 interactions. To view the training progress and various logged metrics (including Lux AI S2 specific metrics like total ice dug, water produced) you can use tensorboard as so. By the end of training you should see that the evaluation episode length increases over time to reach 1000, meaning the agent has learned to dig ice and produce water well enough to survive. This trained agent should also surpass the default rule-based python agent.

```
tensorboard --logdir logs
```

You should see your agent generally learn to dig more ice, produce more water, and during evaluation it will survive longer (the eval/mean_ep_length is higher). Note that RL is incredibly unstable to train and sometimes a bad seed may impact training results. For ideas on how to improve your agent stay tuned for a part 3 tutorial showing how to use invalid action masking and more tricks to solve complex multi-agent games like Lux AI S2.

## Evaluation

To start evaluating with the CLI tool and eventually submit to the competition, we need to save our best model (stored in <log_path>/models/best_model.zip) to the root directory. Alternatively you can modify `MODEL_WEIGHTS_RELATIVE_PATH` in agent.py to point to where the model file is. If you ran the training script above it will save the trained agent to `logs/exp_1/models/best_model.zip`.

Once that is setup, you can test and watch your trained agent on the nice HTML visualizer by running the following

```
luxai-s2 main.py main.py --out=replay.html
```

Open up `replay.html` and you can look at what your agent is doing. If training was succesful, you should notice it picking up power, digging ice, and transferring it back to the factory.


## Submitting to Kaggle

To submit your trained agent, first create a .tar.gz file

```
tar -cvzf submission.tar.gz *
```

and submit that to the competition. Make sure that `MODEL_WEIGHTS_RELATIVE_PATH` is pointing to a .zip file in your folder or else the agent won't run.

## Tips for Improving your Agent

This tutorial agent will train a policy that can efficiently control a single heavy robot that learns to pickup power, constantly dig ice, and transfer ice back to the factory and survive the full 1000 turns in the game. A simple improvement would be to add lichen planting to the action space / controller or program it directly as a rule in the agent.py file, allowing you to score points by the end of the game as well as generate more power.

Another easy idea is to modify the `agent.py` code so that you spawn multiple factories and multiple heavy robots, and simply run the trained policy on each heavy robot.


If you want to look into more scalable solutions, it's critical to first figure out how to model multiple units at once. This kit shows you how to control a single heavy robot effectively but not multiple. Another thing to consider is what observations and features would be the most useful. Finally, you can always try and develop a more complex action controller in addition to developing better reward functions.

If you feel you are experienced enough, you can take a look at [last season's winning solution by team Toad Brigade](https://www.kaggle.com/competitions/lux-ai-2021/discussion/294993) or [our paper: Emergent collective intelligence from massive-agent cooperation and competition](https://arxiv.org/abs/2301.01609) which show how to use convolutional neural nets and various other techniques (e.g. invalid action masking) to control a massive number of units at once.