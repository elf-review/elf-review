import os
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoImageProcessor, ResNetForImageClassification
from Utils.config import *

'''
model_original_folder           = "model_original/"
model_compressed_folder         = "model_compressed/"
model_decompressed_folder       = "model_decompressed/"

dup_layer_folder                = model_compressed_folder+"dup_layer_folder/"
model_elves_compression         = model_compressed_folder+"elves_compression/"
'''

def model_downloading(model_name_list):
    for model_name in model_name_list:
        model_downloading_fun(model_name)

def model_downloading_fun(model_name):
    model_foler = model_original_folder+model_name+"/"
    folder_making_fun(model_foler)
    model_path = model_foler+"pytorch_model.bin"
    if not os.path.exists(model_path):
        if model_name == "microsoft_resnet-50":
            model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        else:
            #tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        torch.save(model.state_dict(), model_path)

def folder_making_fun(folder):
    if not os.path.exists(os.path.dirname(folder)):
        #print("Making dir:", folder)
        os.makedirs(os.path.dirname(folder))
