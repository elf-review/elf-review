import time
import pickle
import os
import hashlib
import math
import numpy as np
import torch
import shutil
import struct
import csv
import subprocess
from collections import OrderedDict
from tqdm import tqdm
import tarfile
from queue import PriorityQueue
import zstandard as zstd
import gzip
from Utils.utils import *
from Evaluation.evaluation import *
from main import main

def evaluation_main():
    Elves_status_check(model_name_list)
    print("\n~~~~~~~~~~~~~~~~~~~~ Evaluation Reproducing ~~~~~~~~~~~~~~~~~~~~")
    evaluation(model_name_list)
    print("\n~~~~~~~~~~~~~~~~~~~~ Compression Evaluation Summary ~~~~~~~~~~~~~~~~~~~~~~~")
    compression_summary(model_name_list)



# For evaluation functions below
def Elves_status_check(model_name_list):
    for model_name in model_name_list:
        model_elves_path = model_elves_compression+model_name+"/pytorch_model.tar.zst"
        if not os.path.exists(model_elves_path):
            print("~~~~~~~~~~ Performing ELVES Compression Reproduction ~~~~~~~~~~")
            main()
            break


def compression_summary(model_name_list):

    model_hd_size_dict = get_repeated_hash_layer_size()
    #delete_folder(dup_layer_folder)
    cnt = 0
    org_total = 0
    elves_total = 0
    gzip_total = 0
    zstd_total = 0
    chimp_total = 0
    gorilla_total = 0
    
    chimp_cmp_name_bits_folder = "Evaluation/chimp/chimp_para_bits/"
    gorilla_cmp_name_bits_folder = "Evaluation/chimp/gorilla_para_bits/"
    for model_name in model_name_list:
        cnt += 1
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_size_org = os.path.getsize(model_path)
        org_total += model_size_org

        model_elves_path = model_elves_compression+model_name+"/pytorch_model.tar.zst"
        model_size_elves = os.path.getsize(model_elves_path) + model_hd_size_dict[model_name]
        elves_total += model_size_elves

        model_gzip_path = model_compressed_folder+"Gzip/"+model_name+"/pytorch_model.bin.gz"
        model_size_gzip = os.path.getsize(model_gzip_path)
        gzip_total += model_size_gzip

        model_zstd_path = model_compressed_folder+"zstd/"+model_name+"/pytorch_model.bin.zst"
        model_size_zstd = os.path.getsize(model_zstd_path)
        zstd_total += model_size_zstd
        
        model_chimp_file_path = chimp_cmp_name_bits_folder+model_name+".csv.gz.csv"
        model_size_chimp = model_size_org
        if not os.path.exists(model_chimp_file_path):
            print(model_name, "Not Compressable by Chimp.")
        else:
            with open(model_chimp_file_path, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    bits = float(row[0])
                    break
            model_size_chimp = rounding(model_size_org * (bits/32))
        chimp_total += model_size_chimp

        model_gorilla_file_path = gorilla_cmp_name_bits_folder+model_name+".csv.gz.csv"
        model_size_gorilla = model_size_org
        if not os.path.exists(model_gorilla_file_path):
            print(model_name, "Not Compressable by Gorilla.")
        else:
            with open(model_gorilla_file_path, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    bits = float(row[0])
                    break
            model_size_gorilla = rounding(model_size_org * (bits/32))
        gorilla_total += model_size_gorilla

        print("\n", "~~~~~", model_name, "~~~~~")
        #print("Original Size:", rounding(model_size_org/MB), "MB.")
        print("Gzip    Compression Ratio:", rounding(model_size_org/model_size_gzip))
        print("zstd    Compression Ratio:", rounding(model_size_org/model_size_zstd))
        print("Chimp   Compression Ratio:", rounding(model_size_org/model_size_chimp))
        print("Gorilla Compression Ratio:", rounding(model_size_org/model_size_gorilla))
        print("ELVES   Compression Ratio:", rounding(model_size_org/model_size_elves))
 
    delete_folder(chimp_cmp_name_bits_folder)
    delete_folder(gorilla_cmp_name_bits_folder)
    delete_folder("Evaluation/chimp/target")

    print("\n\n~~~~~~~~~~ Overall Compression Ratio: ~~~~~~~~~~")
    print("Gzip    : ", rounding(org_total/gzip_total))
    print("zstd    : ", rounding(org_total/zstd_total))
    print("Chimp   : ", rounding(org_total/chimp_total))
    print("Gorilla : ", rounding(org_total/gorilla_total))
    print("ELVES   : ", rounding(org_total/elves_total))

def get_repeated_hash_layer_size():
    layer_hash_repeat_value_set_file = dup_layer_folder + "hash_layer_repeat_set.pkl"
    layer_hash_repeat_value_set = unseal_pickle(layer_hash_repeat_value_set_file)

    hash_dedup_size_total = 0
    layer_hash_repeat_value_amortized_size_dict = dict()
    for hash_value in layer_hash_repeat_value_set:
        layer_size = os.path.getsize(dup_layer_folder+hash_value+".pkl")
        layer_hash_repeat_value_amortized_size_dict[hash_value] = {"layer_size":layer_size, "cnt":0}
        hash_dedup_size_total += layer_size

    for model_name in model_name_list:
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        try:
            model = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(model_path, "model load unseccessful!")
            sys.exit()

        same_storage_dict = get_shared_storage_tensor_dict(model)
        for layer_name in model:
            if layer_name in same_storage_dict:
                continue
            layer = model[layer_name]
            weights_numpy = layer.numpy()
            md5_hash = hashlib.md5(weights_numpy.tobytes()).hexdigest()
            if md5_hash in layer_hash_repeat_value_amortized_size_dict:
                layer_hash_repeat_value_amortized_size_dict[md5_hash]["cnt"] += 1
    for layer_hash in layer_hash_repeat_value_amortized_size_dict:
        layer_hash_repeat_value_amortized_size_dict[layer_hash]["avg_size"] = layer_hash_repeat_value_amortized_size_dict[hash_value]["layer_size"] / layer_hash_repeat_value_amortized_size_dict[hash_value]["cnt"]
    model_hd_size_dict = dict()
    for model_name in model_name_list:
        model_hd_size_dict[model_name] = 0
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        try:
            model = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(model_path, "model load unseccessful!")
            sys.exit()

        same_storage_dict = get_shared_storage_tensor_dict(model)
        for layer_name in model:
            if layer_name in same_storage_dict:
                continue
            layer = model[layer_name]
            weights_numpy = layer.numpy()
            md5_hash = hashlib.md5(weights_numpy.tobytes()).hexdigest()
            if md5_hash in layer_hash_repeat_value_amortized_size_dict:
                model_hd_size_dict[model_name] += layer_hash_repeat_value_amortized_size_dict[md5_hash]["avg_size"]
    return model_hd_size_dict


if __name__=='__main__':
    total_start = time.time()
    evaluation_main()
    total_end = time.time()
    print("\nTotal running time:", round((total_end - total_start)/60,2), "mins")

