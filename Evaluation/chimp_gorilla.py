import pickle
import os
import math
import numpy as np
import torch
import shutil
import struct
import csv
import subprocess
from tqdm import tqdm
import tarfile
import zstandard as zstd
import gzip
import sys
sys.path.append("..")
from Utils.utils import *
from Utils.config import *


def chimp_gorilla(model_name_list):
    

    chimp_folder = model_compressed_folder+"Chimp/"
    folder_making_fun(chimp_folder)
    chimp_weights_folder = chimp_folder+"model_preprocessing_for_chimp_gorilla/"
    folder_making_fun(chimp_weights_folder)
    weights_gz_folder = "chimp/src/test/resources/"
    folder_making_fun(chimp_weights_folder)

    
    cnt = 0
    limit_num = 10000*1000
    for model_name in model_name_list:
        cnt += 1
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        weights_gz_file_path = weights_gz_folder+model_name+".csv.gz"
        if os.path.exists(weights_gz_file_path):
            print(cnt, model_path, model_name, "preprocessed.")
            continue
        model_size = os.path.getsize(model_path)/MB
        print(cnt, model_name, "preprocessing...")
        model_weights_folder_individual = chimp_weights_folder+model_name+"/"
        folder_making_fun(model_weights_folder_individual)
        try:
            model = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print("model load by torch failed.")
            sys.exit()
        same_storage_dict = get_shared_storage_tensor_dict(model)
        model_weights_flatten_f32 = list()
        
        layer_cnt_num = len(model)
        layer_cnt = 0
        para_cnt_num = 0
        for layer_name in model:
            layer_cnt += 1
            if layer_name in same_storage_dict:
                continue
            layer = model[layer_name]
            weights_numpy = layer.numpy().flatten()
            if layer.dtype == torch.float32:
                para_cnt_num += len(weights_numpy)
            if para_cnt_num >= limit_num:
                layer_cnt_num = layer_cnt
                #print("model parameters too much for Chimp, preprocessing...")
                break

        pbar = tqdm(total=layer_cnt_num)
        layer_cnt = 0
        for layer_name in model:
            pbar.update(1)
            layer_cnt += 1
            if layer_name in same_storage_dict:
                continue
            layer = model[layer_name]
            weights_numpy = layer.numpy().flatten()
            layer_dtype = layer.dtype
            if layer_dtype == torch.float32:
                for para in weights_numpy:
                    model_weights_flatten_f32.append(para)
            if layer_cnt >= layer_cnt_num:
                break
        pbar.close()
        print("Chimp Data Preprocessing...")
        if len(model_weights_flatten_f32) > 0:
            model_weights_flatten_f32 = np.array(model_weights_flatten_f32)
            para_len = len(model_weights_flatten_f32)
            model_weights_flatten_file = model_weights_folder_individual + "f32_" + str(para_len)+".csv"
            with open(model_weights_flatten_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for para in model_weights_flatten_f32:
                    writer.writerow([None,None,para])
        #else:
            #print("No Float32 Parameters.")

        for file_name in os.listdir(model_weights_folder_individual):
            if ".csv" in file_name:
                #print(file_name, "EXISTS..")
                weight_file_path = os.path.join(model_weights_folder_individual, file_name)
                '''
                with open(weight_file_path, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    row_cnt = 0
                    for row in reader:
                        print(row_cnt, row)
                        row_cnt += 1
                        if row_cnt >= 10:
                            break
                '''
                file_exist_flg = True
                weights_gz_file_path = "Evaluation/chimp/src/test/resources/"+model_name+".csv.gz"
                if os.path.exists(weights_gz_file_path):
                    #print(weights_gz_file_path, "Exists...")
                    continue
                #print("compressing...", weight_file_path, "compressed to:", weights_gz_file_path)
                with open(weight_file_path, "rb") as f_in, gzip.open(weights_gz_file_path, "wb") as f_out:
                    f_out.writelines(f_in)

    os.chdir('Evaluation/chimp')
    cmd = ['mvn', 'test', '-Dtest=TestSinglePrecision']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    while True:
        line = process.stdout.readline()
        if line == '' and process.poll() is not None:
            break
        if line:
            print(line.strip())
   
    for model_name in model_name_list:
        model_gz_file = "src/test/resources/"+model_name+".csv.gz"
        delete_file(model_gz_file)
    

    # Get the current directory
    current_dir = os.getcwd()
    #print("Current dir:", current_dir)
    # Go up two levels
    grandmother_dir = os.path.dirname(os.path.dirname(current_dir))
    #print("Grand Dir:", grandmother_dir)
    # Change directory to the grandmother_dir
    os.chdir(grandmother_dir)
    #print("Current Dir after ch:", os.getcwd())

