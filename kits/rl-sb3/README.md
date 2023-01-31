# Stable Baselines 3 Simple RL Kit

This is a simple stable baselines 3 RL kit based off of part 2 of the [RL tutorial series]()

`train.py` implements a very simple approach to training an RL agent to dig ice and deliver it back to factories to generate water and survive longer.

The general structure of this RL agent is that we use a heuristic policy to handle bidding and factory placement, then train the RL agent on solving the normal phase of the game. The goal of this RL agent is to survive as long as possible without worrying about growing lichen yet.

To use the training code, run `train.py --help` for help and to train an agent run

```
python train.py --n-envs 10 --log-path logs/exp_1  --total-timesteps 3500000 --seed 42
```

which trains an RL agent using the PPO algorithm with 10 parallel environments for 4,000,000 interactions. To view the training progress and various logged metrics (including Lux AI S2 specific metrics like total ice dug, water produced) you can use tensorboard as so.

```
tensorboard --logdir logs
```

You should see your agent generally learn to dig more ice, produce more water, and during evaluation it will survive longer (the eval/mean_ep_length is higher). Note that RL is incredibly unstable to train and sometimes a bad seed may impact training results. For ideas on how to improve your agent stay tuned for a part 3 tutorial showing how to use invalid action masking and more tricks to solve complex multi-agent games like Lux AI S2.


To submit your trained agent, run

```
tar -cvzf submission.tar.gz *
```

## Tips for Improving your Agent

The biggest thing to tackle first would be to figure out how to model multiple units at once. This kit shows you how to control a single heavy robot effectively but not multiple. Another thing to consider is what observations and features would be the most useful. Finally, you can always try and develop a more complex action controller in addition to developing better reward functions.

If you feel you are experienced enough, you can take a look at [last season's winning solution by team Toad Brigade](https://www.kaggle.com/competitions/lux-ai-2021/discussion/294993) or [our paper: Emergent collective intelligence from massive-agent cooperation and competition](https://arxiv.org/abs/2301.01609) which show how to use convolutional neural nets and various other techniques (e.g. invalid action masking) to control a massive number of units at once.