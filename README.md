# Latent Space Safe Sets
Code for 'LS3: Latent Space Safe Sets for Long-Horizon Visuomotor Control of Iterative Tasks'

[Read the paper](https://arxiv.org/abs/2107.04775)

### Bibtex

```
@inproceedings{LS3,
    title={LS3: Latent Space Safe Sets for Long-Horizon Visuomotor Control of Sparse Reward Iterative Tasks},
    author={Wilcox*, Albert and Balakrishna*, Ashwin and Thananjeyan, Brijen and Gonzalez, Joseph E. and Goldberg, Ken},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2021},
    organization={PMLR}
}
```

## Installation

Create a virtual environment with python 3.7 and activate it:

```virtualenv --python=PATH_TO_PYTHON3.7 venv```

```source venv/bin/activate```

From the latent-safe-sets directory, install this package and most of 
its dependencies using

```pip install -e .```

If you would like to run experiments with the reacher environment, 
install our customized DeepMind Control Suite:

```cd dm_control```

```pip install .```

```cd ../dmc2gym```

```pip install -e .```





## Instructions to run LS3

The algorithm has multiple steps:
1. Collect demo trajectories
2. Train a variational autoencoder based on these trajectories
3. Train dynamics, safe sets, value function, goal estimators, and constraint estimators on top of embeddings
4. Collect more trajectories by rolling out with the learned policy and updating the models trained on top of the VAE

Each of these items has scripts corresponding to it, or you can just run one
command to do everything (described below).

Each time you run a script it automatically generates a logdir in the `outputs`
folder based on the current date and time, which is where it saves logs, plots
and models.

### Collecting Data
To collect data, run 

```python scripts/collect_data.py --env ENV_NAME```

If desired, you can tweak the number of trajectories to collect in `latentsafesets/utils/arg_parser.py`
around line 170. This will save data in `data/ENV_NAME` as json and numpy files

To translate this data into the right format for training the VAE, you'll need to run 

```python scripts/data_to_images.py --env ENV_NAME```

### Training Modules

If you choose to train each module individually there is a script for each
module in the `scripts` folder. For example, to train an autoencoder on your
data from the reacher env you would run

```python scripts/train_encoder.py --env reacher --enc_data_aug```

where the `enc_data_aug` flag tells it to augment the data which is highly recommended.

If you've already trained a module and would like to use it when running a script
you can add a flag pointing to it. For example, to train a value function with an
encoder in `temp/vae.pth` you would run

```python scripts/train_value_function.py --env ENV_NAME --enc_checkpoint temp/vae.pth```

#### IMPORTANT NOTE
These flags are not true by default but they are highly recommended. Why did
I set up the code this way? don't know...

When training encoders add a `--enc_data_aug` flag to enable random data augmentations

When training value functions add a `--val_ensemble` flag to enable ensembles.
(Dynamics uses ensembles by default. Sorry for the inconsistent design).

I will add these flags to every command from here on out for those who copy and 
paste.

### Running MPC Learning

There are two methods of running the MPC learning script.

If you've trained all your modules and would like to run the MPC learning
script, the best way to do it is to move them all into one folder, which I'll denote
wit `FOLDER`, name the 
VAE, value function, safe set, dynamics, constraint function, and goal function
`vae.pth`, `ss.pth`, `dyn.pth`, `constr.pth`, and `gi.pth`, then run 

```python scripts/mpc_learning.py --env ENV_NAME --checkpoint_folder FOLDER --enc_data_aug --dyn_overshooting --val_ensemble```

If you run this script without providing checkpoints for certain modules
it will automatically pretrain them. For example if you've trained a VAE
and would like it to train the other modules and then use those for MPC learning
you can run

```python scripts/mpc_learning.py --env ENV_NAME --vae_checkpoint CHECKPOINT_PATH --enc_data_aug --dyn_overshooting --val_ensemble```

and it will train the other modules. If you haven't trained anything, just run

```python scripts/mpc_learning.py --env ENV_NAME --enc_data_aug --dyn_overshooting --val_ensemble```

and it'll train everything for you.

# Replicating results

After collecting data, run 

```python scripts/train_encoder.py --env ENV_NAME --enc_data_aug```

to train a vae. Then, run any of the following commands:

For the navigation environment:

```python scripts/mpc_learning.py --env spb --val_ensemble --plan_hor 5 --safe_set_bellman_coef 0.3 --enc_checkpoint PATH_TO_VAE```

For the reacher environment:

```python scripts/mpc_learning.py --env reacher --val_ensemble --plan_hor 3 --safe_set_thresh 0.5 --safe_set_bellman_coef 0.3 --enc_checkpoint PATH_TO_VAE```

For the sequential pushing environment:

```python scripts/mpc_learning.py --env push --val_ensemble --plan_hor 3 --safe_set_bellman_coef 0.8 --enc_checkpoint PATH_TO_VAE```
