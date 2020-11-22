#!/usr/bin/env python

import os
import os.path as osp
import argparse
from random import shuffle

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_dir", help="images annotated directory")
    parser.add_argument("data_folder", help="folder to choose data")
    args = parser.parse_args()
    list_images = os.listdir(args.data_dir + args.data_folder)
    shuffle(list_images)
    n = len(list_images)
    if n > 0:
        # os.makedirs(osp.join(args.data_dir, "train.txt"))
        # os.makedirs(osp.join(args.data_dir, "valid.txt"))
        num_train = round(n * 0.85)
        num_valid = round(n * 0.1)
        list_train = list_images[0:num_train]
        list_valid = list_images[num_train+1:num_train+num_valid]
        list_test = list_images[num_train+num_valid+1:n]
        with open(args.data_dir + "train.txt","w") as train_file:
            for x in list_train:
                train_file.write(x + "\n")
        print("save train.txt successfully")
        with open(args.data_dir + "valid.txt","w") as valid_file:
            for x in list_valid:
                valid_file.write(x + "\n")
        print("save valid.txt successfully")
        if args.data_folder == "JPEGImages/":
            with open(args.data_dir + "test.txt","w") as valid_file:
                for x in list_valid:
                    valid_file.write(x + "\n")
            print("save test.txt successfully")
    else:
        print("There are no files in this folder")




if __name__ == "__main__":
    main()