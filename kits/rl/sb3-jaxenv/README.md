# Stable Baselines 3 Simple RL Kit using Jax for Environment Speedup

To get started, we highly recommend using conda (or the faster mamba) to set up an environment to manage packages

```
conda env create -f environment.yml
conda activate luxai_s2_jax
```

Then you need to install jax. We recommend following the instructions on the [Jax repository](https://github.com/google/jax#installation) for installing the CUDA powered version. If you do not have access or plan to use a GPU, we don't recommend using the Jax based environment as it will not provide a significant speedup. 