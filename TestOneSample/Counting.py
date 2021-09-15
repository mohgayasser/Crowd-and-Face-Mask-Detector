import os
import numpy as np 
from TestOneSample.main_process import main
from TestOneSample.IOtools import txt_write


def Count():
    opt = dict()

    max_num_list = {0: 22, 1: 7}

    # step1: Create root path for dataset
    opt['num_workers'] = 0

    opt['IF_savemem_test'] = False
    opt['test_batch_size'] = 1

    # --Network settinng
    opt['psize'], opt['pstride'] = 64, 64

    # -- start testing
    set_len = 1

    for ti in range(set_len):
        opt['trained_model_path'] = 'TestOneSample/Model/SHA'
        opt['root_dir'] = os.path.join(r'TestOneSample/data')
        # -- set the max number and partition
        opt['max_num'] = max_num_list[ti]
        opt['step'] = 0.5
        main(opt)