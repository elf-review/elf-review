import pickle
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from Utils.utils import *
from Utils.config import *

def exponential_dedup(model_weights_path_list):
    cnt = 0
    for model_weights_file in model_weights_path_list:
        cnt += 1
        model_name = model_weights_file.split('/')[2]

        # get the model file folder path
        weights_dtype = model_weights_file.split('/')[-1][:3]
        model_file_folder = get_folder_path_for_distance(model_weights_file)+"exponential_dedup/"+weights_dtype+"/"
        #print('\n', cnt, model_file_folder)

        over_para_list_file     = model_file_folder+"exponential_over_para_file.pkl"
        over_position_list_file = model_file_folder+"exponential_over_position_file.pkl"
        within_para_str_file    = model_file_folder+"exponential_within_str_left_file.pkl"
        within_pata_list_file   = model_file_folder+"exponential_within_para_file.pkl"

        if weights_dtype == "f16":
            weights_dtype_np = np.float16
        elif weights_dtype == "f32":
            weights_dtype_np = np.float32
        elif weights_dtype == "f64":
            weights_dtype_np = np.float64
        else:
            print("~~~~ Non clear dtype ~~~~")
            sys.exit()

        model_weights = unseal_pickle(model_weights_file)

        if not os.path.exists(os.path.dirname(model_file_folder)):
            #print("Making dir:", model_file_folder)
            os.makedirs(os.path.dirname(model_file_folder))
        elif os.path.exists(over_para_list_file) and os.path.exists(over_position_list_file) and os.path.exists(within_para_str_file) and os.path.exists(within_pata_list_file):
            #print("~~~~~~~~", model_file_folder, "exists. ~~~~~~~~~~")
            #print(over_para_list_file, "exists.")
            #print(over_position_list_file, "exists.")
            #print(within_para_str_file, "exists.")
            #print(within_pata_list_file, "exists.")
            org_file_size = os.path.getsize(model_weights_file)
            total_file_size = 0
            total_file_size += os.path.getsize(over_para_list_file)
            total_file_size += os.path.getsize(over_position_list_file)
            total_file_size += os.path.getsize(within_para_str_file)
            total_file_size += os.path.getsize(within_pata_list_file)
            if org_file_size <= total_file_size:
                print("No Storage Saving from ELF.")
            else:
                print("ELF:  org file size:", rounding(org_file_size/MB), "MB. ", "cmp file size:", rounding(total_file_size/MB), "MB. ",  "Compression Ratio:", rounding(org_file_size/total_file_size), "\n")
            continue


        over_para_list = list()
        over_position_list = list()
        within_para_str = str()
        within_para_list = list()

        if weights_dtype == "f16":
            limit_max_abs = 0.999
        elif weights_dtype == "f32":
            limit_max_abs = 0.9999999
        elif weights_dtype == "f64":
            limit_max_abs = 0.99999999999
        else:
            print("~~~~ Non clear dtype ~~~~")
            sys.exit()

        position = 0
        pbar = tqdm(total=len(model_weights))
        for para in model_weights:
            if abs(para) < limit_max_abs:
                within_para_str = exp_encoding(para, within_para_str, within_para_list, weights_dtype_np)
            else:
                over_position_list.append(position)
                over_para_list.append(para)
            position += 1
            pbar.update(1)
        pbar.close()

        over_para_list = np.array(over_para_list, dtype=weights_dtype_np)
        over_position_list = np.array(over_position_list)
        seal_pickle(over_para_list_file, over_para_list)
        seal_pickle(over_position_list_file, over_position_list)

        seal_pickle(within_para_str_file, within_para_str)
        within_para_list = np.array(within_para_list, dtype=np.uint64)
        seal_pickle(within_pata_list_file, within_para_list)

        org_file_size = os.path.getsize(model_weights_file)
        total_file_size = 0
        total_file_size += os.path.getsize(over_para_list_file)
        total_file_size += os.path.getsize(over_position_list_file)
        total_file_size += os.path.getsize(within_para_str_file)
        total_file_size += os.path.getsize(within_pata_list_file)

        if org_file_size <= total_file_size:
            print("No Storage Saving from ELF.")
        else:
            print("ELF:  org file size:", rounding(org_file_size/MB), "MB. ", "cmp file size:", rounding(total_file_size/MB), "MB. ",  "Compression Ratio:", rounding(org_file_size/total_file_size), "\n")

        #weight_list_recovery = weights_recovery_from_exponential(weights_dtype_np, over_para_list, over_position_list, within_para_str, within_para_list)
        #weights_error_comparison(model_weights, weight_list_recovery)

def exp_encoding(para, table2_para_str, table2_para_list, weights_dtype_np):
    flg = '0'
    if para < 0:
        flg = '1'
    para = abs(para)
    para += 1.0
    if weights_dtype_np == np.float16:
        bin_f16 = np.binary_repr(np.float16(para).view(np.uint16), width=16)
        table2_para_str += (bin_f16[6:]+flg)
    elif weights_dtype_np == np.float32:
        #bin_f32 = float_to_bin(para)
        bin_f32 = np.binary_repr(np.float32(para).view(np.uint32), width=32)
        table2_para_str += (bin_f32[9:]+flg)
    elif weights_dtype_np == np.float64:
        bin_f64 = np.binary_repr(np.float64(para).view(np.uint64), width=64)
        table2_para_str += (bin_f64[12:]+flg)
    else:
        print("~~~~ Non clear dtype in expe_encoding ~~~~")
        sys.exit()
    table2_para_str = para_update_bits(table2_para_list, table2_para_str)
    return table2_para_str


def para_update_bits(para_list, para_str):
    while len(para_str) >= 64:
        para_list.append(int(para_str[:64], 2))
        para_str = para_str[64:]
    return para_str

