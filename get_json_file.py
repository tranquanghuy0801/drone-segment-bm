import os 
import shutil
import glob
import json

save_folder = 'Json_Deer_Harry'
list_folder = ['Deer_Photos','Deer_Photo2','Deer_Photos3']

def copy_json_files(list_folder, save_folder):
    count = 1
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for folder in list_folder:
        print('Processing folder ' + folder)
        for file in glob.glob(folder + '/*.json'):
            print('Processing folder ' + folder + ' - File name ' + file)
            # read_file = os.path.join(folder,file)
            write_file = os.path.join(save_folder,str(count) + '.json')
            shutil.copy(file,write_file)
            count = count + 1

# Processing json file 
count_labels = [0, 0, 0, 0]

# copy_json_files(list_folder,save_folder)

for file in glob.glob(save_folder + '/*.json'):
    with open(file, 'r') as f:
        label_json = json.load(f)
        for i in label_json['shapes']:
            if i['label'] == 'tree' or i['label'] == 'Trees':
                count_labels[0] += 1
            elif i['label'] == 'ground' or i['label'] == 'Ground':
                count_labels[1] += 1
            elif i['label'] == 'water' or i['label'] == 'Other' :
                count_labels[2] += 1
            else:
                count_labels[3] += 1

print(count_labels)



        