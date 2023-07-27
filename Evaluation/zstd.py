import os
import sys
import numpy as np
import tarfile
import zstandard as zstd
sys.path.append("..")
from Utils.utils import *
from Utils.config import *


def zstd_compression(model_name_list):
    zstd_compression_folder = model_compressed_folder+"zstd/"
    folder_making_fun(zstd_compression_folder)
    cnt = 0
    for model_name in model_name_list:
        cnt += 1
        model_name_folder = zstd_compression_folder+model_name+"/"
        folder_making_fun(model_name_folder)
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_zstd_compressed_file_path = model_name_folder+"pytorch_model.bin.zst"
        if os.path.exists(model_zstd_compressed_file_path):
            #print(cnt, model_name, "zstd Compression Ratio:", os.path.getsize(model_path)/os.path.getsize(model_zstd_compressed_file_path))
            continue
        print("zstandard compressing... ", model_zstd_compressed_file_path)
        compress_file(model_path, model_zstd_compressed_file_path)
        #print(cnt, model_name, "zstd Compression Ratio:", os.path.getsize(model_path)/os.path.getsize(model_zstd_compressed_file_path))

def compress_file(input_path, output_path, compression_level=3):
    cctx = zstd.ZstdCompressor(level=compression_level)
    with open(input_path, 'rb') as input_file:
        with open(output_path, 'wb') as output_file:
            compressor = cctx.stream_writer(output_file)
            for chunk in iter(lambda: input_file.read(8192), b''):
                compressor.write(chunk)
            compressor.flush(zstd.FLUSH_FRAME)
            compressor.close()
