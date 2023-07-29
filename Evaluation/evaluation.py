
from Evaluation.gzip import *
from Evaluation.zstd import *
from Evaluation.chimp_gorilla import chimp_gorilla

def evaluation(model_name_list):
    print("\n~~~~~ Gzip ~~~~~")
    gzip_compression(model_name_list)
    print("\n~~~~~ zstandard ~~~~~")
    zstd_compression(model_name_list)
    print("\n~~~~~ Chimp and Gorilla ~~~~~")
    chimp_gorilla(model_name_list)
