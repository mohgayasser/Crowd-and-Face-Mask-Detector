import os
import numpy as np 
from Testing.main_process import main
from Testing.IOtools import txt_write


def Count():

    opt = dict()

    model_list = {0:'Testing/Model/SHA'}
    max_num_list = {0:22,1:7}

    # step1: Create root path for dataset
    opt['num_workers'] = 0

    opt['IF_savemem_test'] = False
    opt['test_batch_size'] = 1

    # --Network settinng    
    opt['psize'],opt['pstride'] = 64,64

    
    # -- start testing
    set_len =1

    for ti in range(set_len):
      opt['trained_model_path'] = model_list[ti]
      opt['root_dir'] = os.path.join(r'Testing/data')
      #-- set the max number and partition
      opt['max_num'] = max_num_list[ti]  
      partition_method = {0:'one_linear',1:'two_linear'}
      opt['partition'] = partition_method[1]
      opt['step'] = 0.5

        #print('=='*36)
        #print('Begin to test for %s' %(dataset_list[ti]) )
      main(opt)