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


'''
def exponent_decompression(model_name, model_decmp_folder, model_decmp_path):
    #print("\n", model_name," model_recovery_from_exponential...")
    model_recovered_folder = model_decmp_folder
    model_recovered_path = model_decmp_path
    model_original_path = model_original_folder+model_name+"/pytorch_model.bin"
    model_structure_file_path = model_elves_compression+model_name+"/model_structure.pkl"

    model_weights_flatten_f16 = list()
    model_weights_flatten_f32 = list()
    model_weights_flatten_f64 = list()
    model_structure = OrderedDict()

    model_cmp_folder = os.path.join(model_elves_compression, model_name)
    for model_item in os.listdir(model_cmp_folder):
        model_item_path = os.path.join(model_cmp_folder, model_item)
        if model_item == "exponential_dedup":
            for model_float_weights_folder in os.listdir(model_item_path):
                model_float_weights_folder_path = os.path.join(model_item_path, model_float_weights_folder)
                for model_float_weights_exponential_file in os.listdir(model_float_weights_folder_path):
                    exponential_file_path = os.path.join(model_float_weights_folder_path, model_float_weights_exponential_file)
                    if model_float_weights_exponential_file == "exponential_over_para_file.pkl":
                        over_para_list = unseal_pickle(exponential_file_path)
                    elif model_float_weights_exponential_file == "exponential_over_position_file.pkl":
                        over_position_list = unseal_pickle(exponential_file_path)
                    elif model_float_weights_exponential_file == "exponential_within_para_file.pkl":
                        within_para_list = unseal_pickle(exponential_file_path)
                    elif model_float_weights_exponential_file == "exponential_within_str_left_file.pkl":
                        within_para_str = unseal_pickle(exponential_file_path)
                    else:
                        print("Error file type in model_recovery_from_exponential.")
                        sys.exit()
                if model_float_weights_folder == "f16":
                    model_weights_flatten_f16 = weights_recovery_from_exponential(np.float16, over_para_list, over_position_list, within_para_str, within_para_list)
                elif model_float_weights_folder == "f32":
                    model_weights_flatten_f32 = weights_recovery_from_exponential(np.float32, over_para_list, over_position_list, within_para_str, within_para_list)
                elif model_float_weights_folder == "f64":
                    model_weights_flatten_f64 = weights_recovery_from_exponential(np.float64, over_para_list, over_position_list, within_para_str, within_para_list)
                else:
                    print("Error folder in model_recovery_from_exponential for fxx folder.")
                    sys.exit()
                #print(model_float_weights_folder_path)
        elif model_item == "model_structure.pkl":
            model_structure = unseal_pickle(model_item_path)
    #print("len model_structure:", len(model_structure))
    #print("len model_weights_flatten_f16:", len(model_weights_flatten_f16))
    #print("len model_weights_flatten_f32:", len(model_weights_flatten_f32))
    #print("len model_weights_flatten_f64:", len(model_weights_flatten_f64), "\n")

    #model_recovered = model_recovery_structure_weights(model_item_path, model_structure, model_weights_flatten_f16, model_weights_flatten_f32, model_weights_flatten_f64)
    model_recovered = model_recovery_structure_weights(model_cmp_folder, model_structure, model_weights_flatten_f16, model_weights_flatten_f32, model_weights_flatten_f64)
    torch.save(model_recovered, model_recovered_path)
    #model_comparison_exponential(model_original_path, model_recovered_path, model_structure_file_path)
    print("Decompressed Model Saved.\n")


def weights_recovery_from_exponential(weights_dtype_np, over_para_list, over_position_list, within_para_str, within_para_list):
    #model_para_list_recover = np.array(list(), dtype=weights_dtype_np)
    model_para_list_recover = list()
    exp_decoding(weights_dtype_np, within_para_list, within_para_str, model_para_list_recover)
    model_para_list_recover = para_decoding(over_position_list, over_para_list, model_para_list_recover)
    model_para_list_recover = weights_dtype_np(model_para_list_recover)
    return model_para_list_recover

def exp_decoding(weights_dtype_np, table2_para_list, table2_para_str_save, para_list):
    #print("~~~~ Exponential Decoding Starts ~~~~")
    table2_para_str = ""
    pbar = tqdm(total=len(table2_para_list))
    for num in table2_para_list:
        #print("num in within_para_list:", num)
        # convert int64 into binary '064b'
        table2_para_str += format(num, '064b')
        table2_para_str = bits_update_para(weights_dtype_np, para_list, table2_para_str)
        pbar.update(1)
    pbar.close()
    #print("len table2_para_str:", len(table2_para_str), " len table2_para_str_save:", len(table2_para_str_save))
    table2_para_str += table2_para_str_save
    bit_str_left = bits_update_para(weights_dtype_np, para_list, table2_para_str)

    if len(bit_str_left) >= 1:
        print("~~~~~~~~ Bit Str Left!!! ~~~~~~~~")
        sys.exit()
    #print("~~~~ Exponential Decoding Completed ~~~~")

def bits_update_para(weights_dtype_np, para_list, bit_str):
    if weights_dtype_np == np.float16:
        decimal_bit_sign = 10 + 1
        sign_exponential_str = "0"+"01111"
    elif weights_dtype_np == np.float32:
        decimal_bit_sign = 23 + 1
        sign_exponential_str = "0"+"01111111"
    elif weights_dtype_np == np.float64:
        decimal_bit_sign = 52 + 1
        sign_exponential_str = "0"+"01111111111"
    else:
        print("~~~~ Non clear dtype in bits_update_para ~~~~")
        sys.exit()

    while len(bit_str) >= decimal_bit_sign:
        sign = bit_str[decimal_bit_sign-1]
        conv_str = sign_exponential_str + bit_str[:decimal_bit_sign-1]
        bit_str = bit_str[decimal_bit_sign:]
        para = binary_to_float(conv_str, weights_dtype_np)
        para = para - 1.0
        if sign == '1':
            para = weights_dtype_np(para*(-1))
        para_list.append(para)
    return bit_str

def binary_to_float(binary_representation, weights_dtype_np):
    if weights_dtype_np == np.float16:
        pack_arg = "H"
    elif weights_dtype_np == np.float32:
        pack_arg = "I"
    elif weights_dtype_np == np.float64:
        pack_arg = "L"
    else:
        print("~~~~ Non clear dtype in binary_to_float ~~~~")
        sys.exit()
    buffer_pack = struct.pack(pack_arg, int(binary_representation, 2))
    float_value = np.frombuffer(buffer_pack, dtype=weights_dtype_np)[0]
    return float_value

def para_decoding(table1_pos, table1_para, para_list_recover):
    #print("~~~~ para_decoding starts ~~~~")

    #print("melloc for para_list_recovered...")
    para_list_recover_new = [0] * (len(table1_pos)+len(para_list_recover))
    pbar = tqdm(total=len(table1_pos))
    for i in range(len(table1_pos)):
        if abs(table1_para[i]) < 0.99:
            print(i, table1_para[i], "in the uncompressed table/")
            sys.exit()
        para_list_recover_new[table1_pos[i]] = table1_para[i]
        pbar.update(1)
    pbar.close()
    cnt = 0
    pbar = tqdm(total=len(para_list_recover_new))
    for i in range(len(para_list_recover_new)):
        # all the 0s are shold be replaced by para_list_recover items
        if abs(para_list_recover_new[i]) < 0.001:
            para_list_recover_new[i] = para_list_recover[cnt]
            cnt += 1
        pbar.update(1)
    pbar.close()
    if cnt != len(para_list_recover):
        print("para_list_recover is not done.")
        sys.exit()
    #print("~~~~ para_decoding finished ~~~~")
    return para_list_recover_new

def model_recovery_structure_weights(model_cmp_folder, model_structure, model_weights_flatten_f16, model_weights_flatten_f32, model_weights_flatten_f64):
    model_recovered = OrderedDict()
    index_f16 = 0
    index_f32 = 0
    index_f64 = 0
    for layer_name in model_structure:
        layer_info_list = model_structure[layer_name]
        if layer_info_list[0] == 0:
            model_recovered[layer_name] = model_recovered[layer_info_list[1]]
            #print("0, inside sharing tensor.")
        elif layer_info_list[0] == 1:
            layer_hashed_file = dup_layer_folder+layer_info_list[1]+".pkl"
            layer_hashed = unseal_pickle(layer_hashed_file)
            model_recovered[layer_name] = torch.from_numpy(layer_hashed.reshape(layer_info_list[2]))
            #print("1, layer hashed tensor.")
        elif layer_info_list[0] == 2:
            non_fl_layer_file = model_cmp_folder+"/non_fl_layers/"+layer_name+".pkl"
            non_fl_layer = unseal_pickle(non_fl_layer_file)
            model_recovered[layer_name] = torch.from_numpy(non_fl_layer.reshape(layer_info_list[1]))
            #print("2, non float weights tensor.")
        else:
            weight_num = reshape_to_flat(layer_info_list[1])
            if layer_info_list[0] == 16:
                model_recovered[layer_name] = torch.from_numpy(np.array(model_weights_flatten_f16[index_f16:index_f16+weight_num]).reshape(layer_info_list[1]))
                index_f16 += weight_num
                #print("16, float16 weights tensor.")
            elif layer_info_list[0] == 32:
                model_recovered[layer_name] = torch.from_numpy(np.array(model_weights_flatten_f32[index_f32:index_f32+weight_num]).reshape(layer_info_list[1]))
                index_f32 += weight_num
                #print("32, float32 weights tensor.")
            elif layer_info_list[0] == 64:
                model_recovered[layer_name] = torch.from_numpy(np.array(model_weights_flatten_f64[index_f64:index_f64+weight_num]).reshape(layer_info_list[1]))
                index_f64 += weight_num
                #print("64, float64 weights tensor.")
            else:
                print("error with the structure code.")
    return model_recovered

def model_comparison_exponential(model_path, model_recovered_path, model_structure_file_path):
    print("~~~ Decompressed Model Validating... ~~~")
    print("original path:", model_path, os.path.getsize(model_path)/MB)
    print("recovery path:", model_recovered_path, os.path.getsize(model_recovered_path)/MB)
    print("model_structure_file_path:", model_structure_file_path)

    model_comparison_shape(model_path, model_recovered_path, model_structure_file_path)

    model = model_loading_fun(model_path)
    model_recovered = model_loading_fun(model_recovered_path)

    #print("\nforward comparision")
    for layer_name in model:
        weights_np_flt = model[layer_name].numpy().flatten()
        weights_np_flt_recovered = model_recovered[layer_name].numpy().flatten()
        weights_error_comparison(weights_np_flt, weights_np_flt_recovered)
        break
    #print("\nbackward comparision")
    for layer_name in reversed(model_recovered):
        weights_np_flt = model[layer_name].numpy().flatten()
        weights_np_flt_recovered = model_recovered[layer_name].numpy().flatten()
        weights_error_comparison(weights_np_flt, weights_np_flt_recovered)
        break

def weights_error_comparison(model_para_list, model_para_list_recover):
    #print("~~~~ encoding vs decoding ~~~~")
    #print("original length:", len(model_para_list), "decoding length:", len(model_para_list_recover))
    if len(model_para_list) != len(model_para_list_recover):
        print("Model Parameter Number Mismatch!!!!!")
        sys.exit()

    #print("~~~~ err info calculation starts ~~~~")
    total_err = 0
    same_para_num = 0
    err_max = PriorityQueue()
    pbar = tqdm(total=len(model_para_list))
    for i in range(len(model_para_list)):
        pbar.update(1)
        if model_para_list[i] == model_para_list_recover[i]:
            same_para_num += 1
            continue

        if math.isnan(model_para_list[i]):
            print("model_para_list is nan with index:", i)
            sys.exit()
        if math.isnan(model_para_list_recover[i]):
            print("model_para_list_recover is nan with index:", i)
            sys.exit()

        err = abs(model_para_list[i]-model_para_list_recover[i])
        total_err += err
        err_max.put((err, i, model_para_list[i], model_para_list_recover[i]))
        if err_max.qsize() >= 5:
            err_max.get()
    pbar.close()

    
    while not err_max.empty():
        err_max_item = err_max.get()
        print("err_max data =", err_max_item)

    print("total err:", total_err)
    print("same ratio:", same_para_num/len(model_para_list))
    

def reshape_to_flat(arr_shape):
    res = 1
    for item in arr_shape:
        res *= item
    return res

def model_comparison_shape(model_original_path, model_decompressed_path, model_strucutre_path):
    print("~~~~ model_comparison_shape ~~~~")
    model_strucutre = unseal_pickle(model_strucutre_path)
    model = model_loading_fun(model_original_path)
    model_recovered = model_loading_fun(model_decompressed_path)

    shape_flg = True

    print("\nforward comparison...")
    for layer_name in model:
        layer_org_shape = model[layer_name].shape
        layer_decmp_shape = model_recovered[layer_name].shape
        layer_structure = model_strucutre[layer_name]
        #print("layer_org_shape:", layer_org_shape, "    layer_decmp_shape:", layer_decmp_shape, "    layer_structure:", layer_structure)
        if layer_org_shape != layer_decmp_shape:
            model_recovered[layer_name] = model_recovered[layer_name].reshape(layer_org_shape)
            print("\n", layer_name, " layer shape inconsistency!")
            print("layer_org_shape:", layer_org_shape, "    layer_decmp_shape:", layer_decmp_shape, "    layer_structure:", layer_structure)
            print("After reshaping:", model_recovered[layer_name].shape)
            sys.exit()

    print("\nbackward comparison...")
    for layer_name in model_recovered:
        layer_org_shape = model[layer_name].shape
        layer_decmp_shape = model_recovered[layer_name].shape
        layer_structure = model_strucutre[layer_name]
        #print("layer_org_shape:", layer_org_shape, "    layer_decmp_shape:", layer_decmp_shape, "    layer_structure:", layer_structure)
        if layer_org_shape != layer_decmp_shape:
            print(layer_name, " layer shape inconsistency!")
            sys.exit()

    if shape_flg == True:
        print("models have same layer shape.")
        return

    print("Shape currection done.")
'''


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
