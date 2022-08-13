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

In fact, more options are given. The meaning of each argument is given in the help section of the source code `diffusion_gen.py`.

**NOTE**:
1. To generate a PIC diffusion model randomly, set `v_isRandom` and `W_isRandom` to `**True**`. To generate one of them randomly is also acceptable. Note that `v_len` is also an parameter of a PIC model.
2. To use a manually-designed PIC diffusion model, set `v_isRandom` and `W_isRandom` to `**False**`. **Note that in this case, parameter `v` and `W` must be given in the file `PIC_para.py`**.
3. When diffusion model is set to `IC` or `LT`, we recommend to use a larger `num_chains`. Our model need sufficient data to perform better on these two models.
4. When diffusion model is set to `PIC` and you want to generate a PIC diffusion model randomly, please check the file `[dir]/[name]/tmp/tmp_weighted_edges.txt`. It is possible that the weights of egdes in random diffusion model generated are all '0's or all '1's. If such occasion happens, please adjust the `scalar` and `offset`.
