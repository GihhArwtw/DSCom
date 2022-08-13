# DSCom: Data-Driven Self-Adaptive Community-Based Influence Maximization Framework
DSCom for Influence Maximization. 

The main contribution of our work is to extend the original Influence Maximization problem to a relaxed one, with diffusion model unknown.



## Diffusion Dataset Generation

Use `diffusion_gen.py` to generate diffusion dataset. An example is given below. 

```cmd
python diffusion_gen.py --name PIC_test --random_seed 20220812 --model PIC --directed False --num_chains 2500 --scalar 10 --offset -8
```

To use a manually-designed PIC diffusion model, you may use the following instruction.

```cmd
python diffusion_gen.py --name PIC_test --random_seed 20220812 --model PIC --directed False --num_chains 2500 --v_isRandom False --W_isRandom False --scalar 10 --offset -8
```

Note that in this case, **manually-designed PIC parameters `v` and `W` must be given in the file `PIC_para.py`**.

Randomly generate **`v`** only or **`W`** only is also accpetable by setting `v_isRandom` or `W_isRandom` to `True`.



---



In fact, more options are given, i.e. `dir`, `dataset_node`, `dataset_edge`, `window_len`, `v_len`. Please check the help section of the source code `diffusion_gen.py` for their meanings.


**NOTES**:
1. For PIC diffusion models, note that `v_len` is also an parameter of a PIC model.
2. When diffusion model is set to `IC` or `LT`, we recommend to use a larger `num_chains`. Our model need sufficient data to perform better on these two models.
3. When diffusion model is set to `PIC` and you want to generate a PIC diffusion model randomly, please check the file `[dir]/[name]/tmp/tmp_weighted_edges.txt`. It is possible that the weights of egdes in random diffusion model generated are all '0's or all '1's. If such occasion happens, please adjust the `scalar` and `offset`.

-----------------------------

## DSCom

### 1. DSCom Model Definition and Training

Use `DSCom_train.py`. An example is given below.

```cmd
python diffusion_gen.py --name PIC_test --random_seed 20220812 --num_epoch 1000
```

If the former training process come to an unexpected end, you may use the following instruction to continue training.

```cmd
python diffusion_gen.py --name PIC_test --random_seed 20220812 --num_epoch 1000 --continue_train True --pretrained_model ./_experiments/PIC_test/model_best.pth
```

---

In fact, more options are given, i.e. `dir`, `learning_rate`, `dropout`, `num_out_emb`, `neg_spl_ratio`. Please check the help section of the source code `DSCom_train.py` for their meanings.

--------------------------------------

### 2. Seed Selection By DSCom

Use `DSCom_pred.py`. An example is given below.

```cmd
python DSCom_pred.py --name PIC_test --num_seeds 10
```

For a dynamic graph whose previous status is already computed, you may use the following instruction to get the updated result.

```cmd
python DSCom_pred.py --name PIC_test --num_seeds 10 --dynamic True --dynamic_base 10_centroids.npy
```

For ablation studies, you may use the followig instruction.

```cmd
python DSCom_pred.py --name PIC_test --num_seeds 10 --ablation True
```

---

In fact, more options are given, i.e. `dir`, `model_under_dir`, `dscom_model`, `dynamic_base_under_dir`. Please check the help section of the source code `DSCom_pred.py` for their meanings.

--------------------------------------

## Baselines

### IMM & SSA

Codes for IMM and SSA algorithms are given in the branch `comp_algos/IMM` and `comp_algos/SSA`.

You can run them manually, or use the following instruction.

```cmd
python comp.py --name PIC_test --num_seeds 10 --IMM_eps 0.1 --IMM_l 3 --SSA_eps 0.1 --SSA_delta 0.01
```

### MAIM

Please check their original paper [Multiple Agents Reinforcement Learning Based Influence Maximization in Social Network Services
](https://link.springer.com/chapter/10.1007/978-3-030-91431-8_27).

------------------------------------

## Evaluations

You may use the following instruction to evaluate the results.

```cmd
python eval.py --name PIC_test --num_seeds 10
```
