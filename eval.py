import argparse
import os

'''
================================================
                    Arguments
================================================
'''

class EvalOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.inited = False
        return
        
    def init(self):
        self.parser.add_argument('--dir', type=str, default='./_experiments', help='the path where dataset temp files, checkpoints and logs are stored.\n  Must be the same one used when generating diffusion dataset.\n  Defaultly sets to \'./dscom_expr\'.')
        self.parser.add_argument('--name', type=str, default='PIC_test', help='name of the experiment.\n  Must be the same one used when generating diffusion dataset.\n  Defaultly sets to \'PIC_test.\'')
        self.parser.add_argument('--num_seeds', type=int, default=10, help='the number of seeds selected.\n  Defaultly set to 10.')
        
        return
        
    def parse(self, save=True):
        if not self.inited:
            self.init()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('\n--------------------- Options ----------------------')
        for key, value in sorted(args.items()):
            print('%s: %s' % (str(key), str(value)))
        print('----------------------- End ------------------------')

        if not os.path.exists(self.opt.dir):
            raise ValueError("DIR ERROR. The path of dataset does not exist.")
            
        expr_dir = os.path.join(self.opt.dir, self.opt.name)
        if not os.path.exists(expr_dir):
            raise ValueError("NAME ERROR. The path of dataset does not exist.")
        
        tmp_dir = os.path.join(expr_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            raise ValueError("DIR or NAME ERROR. The path of dataset does not exist.")
            
        log_dir = os.path.join(expr_dir, 'log')
        if not os.path.exists(log_dir):
            raise ValueError("DIR or NAME ERROR. LOG missing. The path of dataset does not exist.")
        
        
        return self.opt


'''
============================================
                   MAIN
============================================
'''


if __name__ == "__main__":
    
    opt = EvalOptions().parse()
    
    import numpy as np
    import tools

    path = f"{opt.dir}/{opt.name}/"
    base = tools.read_diff_model(path+"tmp/tmp_diff_model.npy")
    nodes = np.load(path+"tmp/tmp_nodes.npy")
    graph = tools.readGraph_direct(path+"tmp/tmp_weighted_edges.txt")

    seeds_imm = set(np.load(path+f'log/{opt.num_seeds}_imm.npy'))
    seeds_ssa = set(np.load(path+f'log/{opt.num_seeds}_ssa.npy'))
    seeds_mxdg = set(np.load(path+f'log/{opt.num_seeds}_dscom_mxdg.npy'))
    seeds_mxct = set(np.load(path+f'log/{opt.num_seeds}_dscom_mxct.npy'))
    seeds_core = set(np.load(path+f'log/{opt.num_seeds}_dscom_core.npy'))
    seeds_pgRk = set(np.load(path+f'log/{opt.num_seeds}_dscom_pgRk.npy'))

    times_comp = np.load(path+f'log/{opt.num_seeds}_imm_ssa_time.npy')
    times_all = np.load(path+f'log/{opt.num_seeds}_dscom_time.npy')


    inf_imm = tools.compute(graph, base, seeds_imm)
    print("IMM evaluated.")

    inf_ssa = tools.compute(graph, base, seeds_ssa)
    print("SSA evaluated.")

    inf_dscom_mxdg = tools.compute(graph, base, seeds_mxdg)
    print("DSCom-max-degree evaluated.")

    inf_dscom_mxct = tools.compute(graph, base, seeds_mxct)
    print("DSCom-max-centrality evaluated.")

    inf_dscom_core = tools.compute(graph, base, seeds_core)
    print("DSCom-k-core evaluated.")

    inf_dscom_pgRk = tools.compute(graph, base, seeds_pgRk)
    print("DSCom-pageRank evaluated.")
    
    print("============ Influence ===========")
    print("DSCom_Max_Degree:     {:.8f}".format(inf_dscom_mxdg))
    print("DSCom_Max_Centrality: {:.8f}".format(inf_dscom_mxct))
    print("DSCom_k-core:         {:.8f}".format(inf_dscom_core))
    print("DSCom_pageRank:       {:.8f}".format(inf_dscom_pgRk))
    print("----------------------------------")
    print("IMM:                  {:.8f}".format(inf_imm))
    print("SSA:                  {:.8f}".format(inf_ssa))
    print("\n============== Time ==============")
    print("DSCom_Max_Degree:     {}".format(times_all[0]))
    print("DSCom_Max_Centrality: {}".format(times_all[1]))
    print("DSCom_k-core:         {}".format(times_all[2]))
    print("DSCom_pageRank:       {}".format(times_all[3]))
    print("----------------------------------")
    print("IMM:                  {}".format(times_comp[0]))
    print("SSA:                  {}".format(times_comp[1]))


    with open(path+f"log/{opt.num_seeds}_comparisons.txt","w") as file:
        print("============ Influence ===========",file=file)
        print("DSCom_Max_Degree:     {:.8f}".format(inf_dscom_mxdg),file=file)
        print("DSCom_Max_Centrality: {:.8f}".format(inf_dscom_mxct),file=file)
        print("DSCom_k-core:         {:.8f}".format(inf_dscom_core),file=file)
        print("DSCom_pageRank:       {:.8f}".format(inf_dscom_pgRk),file=file)
        print("----------------------------------",file=file)
        print("IMM:                  {:.8f}".format(inf_imm),file=file)
        print("SSA:                  {:.8f}".format(inf_ssa),file=file)
        print("\n============== Time ==============",file=file)
        print("DSCom_Max_Degree:     {}".format(times_all[0]),file=file)
        print("DSCom_Max_Centrality: {}".format(times_all[1]),file=file)
        print("DSCom_k-core:         {}".format(times_all[2]),file=file)
        print("DSCom_pageRank:       {}".format(times_all[3]),file=file)
        print("----------------------------------",file=file)
        print("IMM:                  {}".format(times_comp[0]),file=file)
        print("SSA:                  {}".format(times_comp[1]),file=file)