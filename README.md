# DSCom: Data-Driven Self-Adaptive Community-Based Influence Maximization Framework
DSCom for Influence Maximization. 

The main contribution of our work is to extend the original Influence Maximization problem to a relaxed one, with diffusion model unknown.



## 1. Diffusion Dataset Generation
A simulation of the influence spread in the reality. A diffusion dataset will be generated based on the diffusion model given or generated randomly.

Use `diffusion_gen.py` to generate diffusion dataset.

An example is given below. 

```cmd
python diffusion_gen.py --name PIC_test --random_seed 20220812 --model PIC --directed False --num_chains 2500 --scalar 10 --offset -8
```

To use a manually-designed PIC diffusion model, you may use the following instruction.

```cmd
python diffusion_gen.py --name PIC_test --random_seed 20220812 --model PIC --directed False --num_chains 2500 --v_isRandom False --W_isRandom False --scalar 10 --offset -8
```

Note that in this case, **manually-designed PIC parameters `v` and `W` must be given in the file `PIC_para.py`**.

Randomly generate `**v**` only or `**W**` only is also accpetable by setting `v_isRandom` or `W_isRandom` to `True`.

-------------------------------

In fact, more options are given, i.e. `dir`, `dataset_node`, `dataset_edge`, `window_len`, `v_len`. Please check the help section of the source code `diffusion_gen.py` for their meanings.



**NOTES**:
1. For PIC diffusion models, note that `v_len` is also an parameter of a PIC model.
2. When diffusion model is set to `IC` or `LT`, we recommend to use a larger `num_chains`. Our model need sufficient data to perform better on these two models.
3. When diffusion model is set to `PIC` and you want to generate a PIC diffusion model randomly, please check the file `[dir]/[name]/tmp/tmp_weighted_edges.txt`. It is possible that the weights of egdes in random diffusion model generated are all '0's or all '1's. If such occasion happens, please adjust the `scalar` and `offset`.



## 2. DSCom Model Definition and Training

Use `DSCom_train.py`. An example is given below.

```cmd
python diffusion_gen.py --name PIC_test --random_seed 20220812 --num_epoch 1000
```

If the former training process come to an unexpected end, you may use the following instruction to continue training.

```cmd
python diffusion_gen.py --name PIC_test --random_seed 20220812 --num_epoch 1000 --continue_train True --pretrained_model ./_experiments/PIC_test/model_best.pth
```

-------------------------------

In fact, more options are given, i.e. `dir`, `learning_rate`, `dropout`, `num_out_emb`, `neg_spl_ratio`. Please check the help section of the source code `diffusion_gen.py` for their meanings.

