import os
from tqdm import tqdm
import hashlib
import sys
import torch
import pickle
sys.path.append("..")
from Utils.utils import *
from Utils.config import *

def hash_deduplication(model_name_list):
    layer_hash_repeat_value_set_file = dup_layer_folder+"hash_layer_repeat_set.pkl"
    if os.path.exists(layer_hash_repeat_value_set_file):
        #print("file", layer_hash_repeat_value_set_file, "exists, layer hash deduplication finished.")
        return

    layer_hash_value_set = set()
    layer_hash_repeat_value_set = set()

    cnt = 0
    model_size_total = 0
    model_hash_dedup_total = 0
    total_layer_num = 0
    hash_dedup_num = 0

    for model_name in model_name_list:
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        cnt += 1
        model_size = os.path.getsize(model_path)
        model_size_total += model_size
        print("Model:", model_name, str(round(model_size/MB, 2))+" MB")
        model = model_loading_fun(model_path)
        total_layer_num += len(model)

        same_storage_dict = get_shared_storage_tensor_dict(model)

        pbar = tqdm(total=len(model))
        for layer_name in model:
            pbar.update(1)
            if layer_name in same_storage_dict:
                continue

            layer = model[layer_name]
            weights_numpy = layer.numpy()
            md5_hash = hashlib.md5(weights_numpy.tobytes()).hexdigest()
            if md5_hash not in layer_hash_value_set:
                layer_hash_value_set.add(md5_hash)
                continue
            if md5_hash not in layer_hash_repeat_value_set:
                layer_hash_repeat_value_set.add(md5_hash)
                layer_hash_file = dup_layer_folder+md5_hash+".pkl"
                seal_pickle(layer_hash_file, weights_numpy)
            model_hash_dedup_total += weights_numpy.nbytes
        pbar.close()

    seal_pickle(layer_hash_repeat_value_set_file, layer_hash_repeat_value_set)
    
    
    hash_dedup_num = len(layer_hash_repeat_value_set)
    print("~~~~~ Hash Deduplication (HD) Summary ~~~~~")
    print("total_layer_num :", total_layer_num)
    #print("hash_total_num  :", len(layer_hash_value_set))
    print("hash_dedup_num  :", hash_dedup_num)
    print("model_total_size:", model_size_total/MB, "MB")
    print("hash_dedup_size :", model_hash_dedup_total/MB, "MB")
    print("HD Compression Ratio:", model_size_total/(model_size_total-model_hash_dedup_total))
    

