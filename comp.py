import argparse
import os

'''
================================================
                    Arguments
================================================
'''

def str2bool(v):
    return v.lower() in ('true')

class CompOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.inited = False
        return
        
    def init(self):
        self.parser.add_argument('--dir', type=str, default='./_experiments', help='the path where dataset temp files, checkpoints and logs are stored.\n  Must be the same one used when generating diffusion dataset.\n  Defaultly sets to \'./dscom_expr\'.')
        self.parser.add_argument('--name', type=str, default='PIC_test', help='name of the experiment.\n  Must be the same one used when generating diffusion dataset.\n  Defaultly sets to \'PIC_test.\'')
        
        self.parser.add_argument('--num_seeds', type=int, default=20, help='the number of seeds selected for influence maximization problem.\n  Defaultly set to 20.')
        self.parser.add_argument('--IMM_eps', type=float, default=0.1, help='EPSILON in IMM.\n  Defaultly set to 0.1.')
        self.parser.add_argument('--IMM_l', type=float, default=3, help='l in IMM.\n  Defaultly set to 3.')
        self.parser.add_argument('--SSA_eps', type=float, default=0.1, help='EPSILON in SSA.\n  Defaultly set to 0.1.')
        self.parser.add_argument('--SSA_delta', type=float, default=0.01, help='DELTA in SSA.\n  Defaultly set to 0.01.')

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
            raise ValueError("DIR or NAME ERROR. TMP missing. The path of dataset does not exist.")
        
        log_dir = os.path.join(expr_dir, 'log')
        if not os.path.exists(log_dir):
            raise ValueError("DIR or NAME ERROR. LOG missing. The path of dataset does not exist.")
            
        
        if save:
            file_name = os.path.join(expr_dir, 'comp_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('--------------------- Options ----------------------\n')
                for key, value in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(key), str(value)))
                opt_file.write('----------------------- End ------------------------\n')
        
        return self.opt



if __name__ == '__main__':
    
    opt = CompOptions().parse()
    
    import tools
    import time
    import numpy as np
    
    path = f"{opt.dir}/{opt.name}/"
    base = tools.read_diff_model(path+"tmp/tmp_diff_model.npy")
    graph = tools.readGraph_direct(path+"tmp/tmp_weighted_edges.txt")
    
    
    # for SSA
    print("SSA.")
    ssa_time = time.perf_counter()
    
    if base=='IC-based':
        from comp_algos.SSA.SSA import SSA
        seeds_ssa = SSA(graph, opt.num_seeds, opt.SSA_eps, opt.SSA_delta)
    else:
        from comp_algos.SSA.SSA_LT import SSA_LT
        seeds_ssa = SSA_LT(graph, opt.num_seeds, opt.SSA_eps, opt.SSA_delta)
    
    ssa_time = time.perf_counter() - ssa_time
    
    print("SSA Fin.\n")
    seeds_ssa = list(seeds_ssa)
    seeds_ssa = np.array(seeds_ssa)
    np.save(path+f'log/{opt.num_seeds}_ssa.npy',seeds_ssa)
    
    
    # for IMM
    print("IMM.")
    imm_time = time.perf_counter()
    
    if base=='IC-based':
        from comp_algos.IMM.IMM import IMM
        seeds_imm = IMM(graph, opt.num_seeds, opt.IMM_eps, opt.IMM_l)
    else:
        from comp_algos.IMM.IMM_LT import IMM_LT
        seeds_imm = IMM_LT(graph, opt.num_seeds, opt.IMM_eps, opt.IMM_l)
    
    imm_time = time.perf_counter() - imm_time
    
    print("IMM Fin.")
    seeds_imm = list(seeds_imm)
    seeds_imm = np.array(seeds_imm)
    np.save(path+f'log/{opt.num_seeds}_imm.npy',seeds_imm)
    
    
    seeds_imm = list(seeds_imm)
    seeds_ssa = list(seeds_ssa)
    print("Number of Seeds: {}".format(opt.num_seeds))
    print("\nSeeds:")
    print("IMM: {}".format(seeds_imm))
    print("SSA: {}".format(seeds_ssa))
    print("\nTime:")
    print("IMM: {}".format(imm_time))
    print("SSA: {}".format(ssa_time))
    
    with open(path+f'log/{opt.num_seeds}_imm+ssa.txt',"w") as file:
        print("Number of Seeds: {}".format(opt.num_seeds),file=file)
        print("\nSeeds:",file=file)
        print("IMM: {}".format(seeds_imm),file=file)
        print("SSA: {}".format(seeds_ssa),file=file)
        print("\nTime:",file=file)
        print("IMM: {}".format(imm_time),file=file)
        print("SSA: {}".format(ssa_time),file=file) 
    
    
    # Time Record.
    time_all = np.array([imm_time, ssa_time])
    np.save(path+f'log/{opt.num_seeds}_imm_ssa_time.npy',time_all)
    
    