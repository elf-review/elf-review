import os
import numpy as np
import pickle
from tqdm import tqdm
import sys
sys.path.append("..")
from Utils.utils import *
from Utils.config import *

def distance_reference(model_weights_path_list):
    #model_weights_path_list = get_weigths_path_list(model_elves_compression)

    cnt = 0
    for model_weights_file in model_weights_path_list:
        cnt += 1
        # get the model file folder path
        weights_dtype = model_weights_file.split('/')[-1][:3]
        model_file_folder = get_folder_path_for_distance(model_weights_file)+"distance_reference/"+weights_dtype+"/"

        #print(cnt, model_file_folder)

        distinct_list_file = model_file_folder + "distinct_file.pkl"
        distance_list_file = model_file_folder + "distance_file.pkl"
        distance_str_file  = model_file_folder + "distance_str_left_file.pkl"

        if not os.path.exists(os.path.dirname(model_file_folder)):
            #print("Making dir:", model_file_folder)
            os.makedirs(os.path.dirname(model_file_folder))
        elif os.path.exists(distinct_list_file) and os.path.exists(distance_list_file) and os.path.exists(distance_str_file):
            #print("~~~~~~~~", model_file_folder, "exists. ~~~~~~~~~~")
            #print(distinct_list_file, "exists.")
            #print(distance_list_file, "exists.")
            #print(distance_str_file, "exists.")
            org_file_size = os.path.getsize(model_weights_file)
            total_file_size = 0
            total_file_size += os.path.getsize(distinct_list_file)
            total_file_size += os.path.getsize(distance_list_file)
            total_file_size += os.path.getsize(distance_str_file)
            if org_file_size <= total_file_size:
                print("No Storage Saving from DE.")
            else:
                print("DE:  org file size:", rounding(org_file_size/MB), "MB. ", "cmp file size:", rounding(total_file_size/MB), "MB. ",  "Compression Ratio:", rounding(org_file_size/total_file_size))
            continue

        if weights_dtype == "f16":
            weights_dtype_np = np.float16
        elif weights_dtype == "f32":
            weights_dtype_np = np.float32
        elif weights_dtype == "f64":
            weights_dtype_np = np.float64
        else:
            print("~~~~ Non clear dtype ~~~~")
            sys.exit()

        # Load the binary file
        # but this method only works for size smaller than 5G.
        model_weights = np.fromfile(model_weights_file, dtype=weights_dtype_np)
        
        #model_weights = unseal_pickle(model_weights_file)

        # currently using 5 bits as the encoding bits to represent the distance length
        distinct_list = list()
        distance_list = list()
        distance_str = ""

        para_position_dict = dict()

        position = 0
        pbar = tqdm(total=len(model_weights))
        for para in model_weights:
            if para not in para_position_dict:
                para_position_dict[para] = position
                distinct_list.append(para)
                distance_str += '0'
            else:
                distance = position - para_position_dict[para]
                para_position_dict[para] = position

                distance_bin = bin(distance)[2:]
                distance_bin_len = len(distance_bin)
                distance_bin_len_bin = bin(distance_bin_len)[2:]
                distance_bin_len_bin_len = len(distance_bin_len_bin)

                para_5bit = (5-distance_bin_len_bin_len)*'0'+distance_bin_len_bin
                distance_str += ('1'+para_5bit+distance_bin)
            distance_str = distance_update_bits(distance_list, distance_str)
            position += 1
            pbar.update(1)
        pbar.close()


        #print("\nbefore distinct_list type:", type(distinct_list[0]), distinct_list[:10])
        #print("before distance_list type:", type(distance_list[0]), distance_list[:10])

        distinct_list = np.array(distinct_list, dtype=weights_dtype_np)
        seal_pickle(distinct_list_file, distinct_list)

        distance_list = np.array(distance_list, dtype=np.uint64)
        seal_pickle(distance_list_file, distance_list)
        seal_pickle(distance_str_file, distance_str)

        #print("after distinct_list type:", distinct_list[0].dtype, distinct_list[:10])
        #print("after distance_list type:", distance_list[0].dtype, distance_list[:10], "\n")

        org_file_size = os.path.getsize(model_weights_file)
        total_file_size = 0
        total_file_size += os.path.getsize(distinct_list_file)
        total_file_size += os.path.getsize(distance_list_file)
        total_file_size += os.path.getsize(distance_str_file)

        if org_file_size <= total_file_size:
            print("No Storage Saving from DE.")
        else:
            print("DE:  org file size:", rounding(org_file_size/MB), "MB. ", "cmp file size:", rounding(total_file_size/MB), "MB. ",  "Compression Ratio:", rounding(org_file_size/total_file_size))

def distance_update_bits(distance_list, bit_str):
    while len(bit_str) >= 64:
        distance_list.append(int(bit_str[:64], 2))
        bit_str = bit_str[64:]
    return bit_str

