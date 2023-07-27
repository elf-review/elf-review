
from Evaluation.gzip import *
from Evaluation.zstd import *

def evaluation(model_name_list):
    print("\n~~~~~ Gzip ~~~~~")
    gzip_compression(model_name_list)
    print("\n~~~~~ zstandard ~~~~~")
    zstd_compression(model_name_list)


