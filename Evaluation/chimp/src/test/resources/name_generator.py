import os


folder = "/resources/"
file_list = list()

for file_name in os.listdir(folder):
    if ".csv.gz" in file_name:
        file_name = "\"/"+file_name+"\","
        print(file_name)

