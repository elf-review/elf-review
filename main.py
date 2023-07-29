import time
import pickle
import os
import math
import numpy as np
import torch
import shutil
import struct
from collections import OrderedDict
from tqdm import tqdm
import tarfile
from queue import PriorityQueue
import zstandard as zstd
import gzip
from Utils.utils import *
from Utils.model_downloading import * 
from ELF.hashing_deduplication import *
from ELF.model_structure import *
from ELF.distance_encoding import *
from ELF.exponentless_floating import *
from ELF.decompression import exponent_decompression
from Evaluation.evaluation import *


def main():
    #print("\n~~~~~ Elves Starts ~~~~~")
    #print("\n~~~~~ Folder Making ~~~~~")
    folder_making()
    print("\n~~~~~~~~~~~~~~~~~~~~ Model Downloading ~~~~~~~~~~~~~~~~~~~~")
    model_downloading(model_name_list)
    print("\n\n~~~~~~~~~~~~~~~~~~~~ ELVES Starts ~~~~~~~~~~~~~~~~~~~~")
    print("\n~~~~~ 1. Hash Deduplicating (HD) ~~~~~")
    hash_deduplication(model_name_list)
    #print("\n~~~~~ Model Structures Saving ~~~~~")
    save_model_structure_and_flatten_weights(model_name_list)
    print("\n~~~~~ 2. Distance Encoding (DE) + Exponent-Less Floating (ELF) ~~~~~")
    ELVES(model_elves_compression)
    #print("\n~~~~~~~~~~~~~~~~~~~~ Evaluation Reproducing ~~~~~~~~~~~~~~~~~~~~")
    #evaluation(model_name_list)
    #print("\n~~~~~ Compression Summary ~~~~~")
    #compression_summary(model_name_list)


def compression_summary(model_name_list):
    print("\n~~~~~~~~~~ Compression Evaluation Summary ~~~~~~~~~~")
    model_hd_size_dict = get_repeated_hash_layer_size()
    #delete_folder(dup_layer_folder)
    cnt = 0
    org_total = 0
    elves_total = 0
    gzip_total = 0
    zstd_total = 0
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

        print("\n", "~~~~~", model_name, "~~~~~")
        print("Original Size:", rounding(model_size_org/MB), "MB.")
        print("Gzip Size:    ", rounding(model_size_gzip/MB), "MB,", "Compression Ratio:", rounding(model_size_org/model_size_gzip))
        print("zstd Size:    ", rounding(model_size_zstd/MB), "MB,", "Compression Ratio:", rounding(model_size_org/model_size_zstd))
        print("ELVES Size:   ", rounding(model_size_elves/MB), "MB,", "Compression Ratio:", rounding(model_size_org/model_size_elves))

    print("\n\n~~~~~~~~~~ Overall Compression Ratio: ~~~~~~~~~~")
    print("Gzip: ", rounding(org_total/gzip_total))
    print("zstd: ", rounding(org_total/zstd_total))
    print("ELVES:", rounding(org_total/elves_total))


def ELVES(model_elves_compression):
    model_weights_path_list = get_weigths_path_list(model_elves_compression)
    cnt = 0 
    for model_weights_file in model_weights_path_list:
        cnt += 1
        model_name = model_weights_file.split('/')[2]
        print("Model:", model_name)
        print("~~~ Distance Encoding (DE) ~~~")
        distance_reference([model_weights_file]) 
        print("~~~ Exponent-Less Floating (ELF) ~~~")
        exponential_dedup([model_weights_file])
    
    print("\n~~~~~ 3. zstd Lossless Compressing ~~~~~")
    cnt = 0
    model_decompression_dict = dict()
    for model_name in model_name_list:
        cnt += 1
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_size_org = os.path.getsize(model_path)
        #model_compressed_file = model_elves_compression+model_name+"/pytorch_model.tar.zst"
        print("Model:", model_name, " zstandard compressing for ELVES intermediate outputs...")
        if os.path.exists(model_elves_compression+model_name+"/pytorch_model.tar.zst"):
            #print("\nELVES Compression Overview: Model:", model_name, "Original Size:", rounding(model_size_org/MB), "MB. ", "Compressed Size:", rounding(os.path.getsize(model_compressed_file)/MB), "MB. ", "Compression Ratio:", model_size_org/os.path.getsize(model_compressed_file), "\n")
            continue 

        weights_folder = model_elves_compression+model_name+"/fl_weights/"
        model_size_weights = get_folder_size(weights_folder)
        de_folder = model_elves_compression+model_name+"/distance_reference/"
        model_size_de = get_folder_size(de_folder)
        elf_folder = model_elves_compression+model_name+"/exponential_dedup/"
        model_size_elf = get_folder_size(elf_folder)
        
        if model_size_weights == min(model_size_weights, model_size_de, model_size_elf):
            model_compressed_weights_folder = weights_folder
            delete_folder(de_folder)
            delete_folder(elf_folder)
        elif model_size_de == min(model_size_weights, model_size_de, model_size_elf):
            model_compressed_weights_folder = de_folder
            delete_folder(weights_folder)
            delete_folder(elf_folder)
        elif model_size_elf == min(model_size_weights, model_size_de, model_size_elf):
            model_compressed_weights_folder = elf_folder
            delete_folder(weights_folder)
            delete_folder(de_folder)
        else:
            print("model size calculating error.")
            sys.exit()
        
        source_folder = model_elves_compression+model_name+"/"
        output_file = model_elves_compression+model_name+"/pytorch_model.tar"
        compress_folder(source_folder, output_file)
        model_decompression_dict[model_name] = model_compressed_weights_folder
        
    print("\n~~~~~~~~~~ ELVES Decompressing ~~~~~~~~~~")
    cnt = 0
    for model_name in model_name_list:
        cnt += 1
        print("Model:", model_name, " Decompressing...")
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_decmp_folder = model_decompressed_folder + model_name + "/"
        folder_making_fun(model_decmp_folder)
        model_decmp_path = model_decmp_folder+"pytorch_model_decmp.bin"
        if os.path.exists(model_decmp_path):
            continue
        if model_decompression_dict[model_name] != model_elves_compression+model_name+"/exponential_dedup/":
            shutil.copy(model_path, model_decmp_path)
        else:
            exponent_decompression(model_name, model_decmp_folder, model_decmp_path)

        delete_folder(model_decompression_dict[model_name])
        delete_file(model_elves_compression+model_name+"/model_structure.pkl")
        non_fl_folder = model_elves_compression+model_name+"/non_fl_layers"
        if os.path.exists(non_fl_folder):
            delete_folder(non_fl_folder)
        
        '''
        source_folder = model_compressed_folder+"dup_layer_folder"
        output_file = model_compressed_folder+"dup_layer_folder.tar"
        compress_folder(source_folder, output_file)
        '''
   
    total_org_size = 0
    total_cmp_size = 0
    print("\n\n~~~~~~~~~~~~~~~~~~~~ ELVES Compression Overview ~~~~~~~~~~~~~~~~~~~~")
    cnt = 0
    for model_name in model_name_list:
        cnt += 1
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_size_org = os.path.getsize(model_path)
        total_org_size += model_size_org
        model_compressed_file = model_elves_compression+model_name+"/pytorch_model.tar.zst"
        model_size_cmp = os.path.getsize(model_compressed_file)
        total_cmp_size += model_size_cmp
        print(cnt, "Model:", model_name, "Original Size:", rounding(model_size_org/MB), "MB. ", "Compressed Size:", rounding(os.path.getsize(model_compressed_file)/MB), "MB. ", "Compression Ratio:", rounding(model_size_org/os.path.getsize(model_compressed_file)))
    print("\nOverall Compression Ratio:", rounding(total_org_size/total_cmp_size))


# for compression 
def compress_folder(source_folder, output_file):
    # Create a tar archive of the folder
    with tarfile.open(output_file, 'w') as tar:
        tar.add(source_folder, arcname=os.path.basename(source_folder))

    # Compress the tar archive using zstd
    cctx = zstd.ZstdCompressor(level=3)  # Set the compression level as needed
    with open(output_file, 'rb') as tar_file:
        with open(f"{output_file}.zst", 'wb') as compressed_file:
            compressed_file.write(cctx.compress(tar_file.read()))

    # Remove the uncompressed tar archive
    os.remove(output_file)


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


def get_weigths_path_list(model_cmp_structure_weights_folder):
    #print("~"*4, "model file listing", "~"*4)
    model_weights_file_list = list()
    # Iterate through each file in the folder
    for model_name in os.listdir(model_cmp_structure_weights_folder):
        model_path = os.path.join(model_cmp_structure_weights_folder, model_name)
        fl_weights_flg = 0
        for model_file in os.listdir(model_path):
            if model_file == "fl_weights":
                model_weights_folder = os.path.join(model_path, model_file)
                cnt = 0
                fl_weights_flg = 1
                for model_weights_file in os.listdir(model_weights_folder):
                    model_weights_file_path = os.path.join(model_weights_folder, model_weights_file)
                    model_weights_file_list.append(model_weights_file_path)
                    cnt += 1
                if cnt == 0:
                    print(model_weights_folder, "fl_weights is empty.")
                    sys.exit()
        if fl_weights_flg == 0:
            print(model_path, "no fl_weights folder.")
            sys.exit()
    return model_weights_file_list


def folder_making():
    folder_making_fun(model_original_folder)
    folder_making_fun(model_compressed_folder)
    folder_making_fun(model_decompressed_folder)
    folder_making_fun(dup_layer_folder)


if __name__=='__main__':
    total_start = time.time()
    main()
    total_end = time.time()
    print("\nTotal running time:", round((total_end - total_start)/60,2), "mins")
