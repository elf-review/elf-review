import os
import gzip
import sys
sys.path.append("..")
from Utils.utils import *
from Utils.config import *

def gzip_compression(model_name_list):

    gzip_compression_folder = model_compressed_folder+"Gzip/"
    folder_making_fun(gzip_compression_folder)

    cnt = 0
    for model_name in model_name_list:
        model_name_folder = gzip_compression_folder+model_name+"/"
        folder_making_fun(model_name_folder)
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        cnt += 1
        model_gz_file_path = model_name_folder+"pytorch_model.bin.gz"
        if os.path.exists(model_gz_file_path):
            #print(cnt, model_name, "Gzip Compression Ratio:", os.path.getsize(model_path)/os.path.getsize(model_gz_file_path))
            continue

        print("Gzip compressing... ", model_gz_file_path)
        with open(model_path, "rb") as f_in, gzip.open(model_gz_file_path, "wb") as f_out:
            f_out.writelines(f_in)





