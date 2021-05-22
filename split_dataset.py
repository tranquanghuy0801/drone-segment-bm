#!/usr/bin/env python

import os
import os.path as osp
import argparse
from random import shuffle
from libs import images2chips
import cv2
import glob

def split_segmentation():
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

def split_gan(data_dir):
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument("data_dir", help="Directory contains RGB Images and Labels data")
	args = parser.parse_args()
	save_dir = "gan-data-new"
	print("Start processing")
	if not os.path.exists(os.path.join(args.data_dir, save_dir)):
		os.makedirs(os.path.join(args.data_dir, save_dir))
		list_dirs = ["train", "test", "val"]
		for directory in list_dirs:
			os.makedirs(os.path.join(args.data_dir, save_dir, directory))
			os.makedirs(os.path.join(args.data_dir, save_dir, directory, "images"))
			os.makedirs(os.path.join(args.data_dir, save_dir, directory, "labels"))
	for num in os.listdir(os.path.join(args.data_dir, 'JPEGImages')):
		num = num.split('.')[0]
		list_images = glob.glob(args.data_dir + '/images-patch2/' + num + '-*.png')
		print(list_images)
		shuffle(list_images)
		n = len(list_images)
		num_train = round(n * 0.8)
		num_val = round(n * 0.1)
		list_train = list_images[0:num_train]
		list_val = list_images[num_train+1:num_train+num_val]
		list_test = list_images[num_train+num_val+1:n]
		print("Processing Train folder")
		for img_train in list_train:
			img_train = img_train.split('/')
			img_train = img_train[len(img_train)-1]
			img = cv2.imread(os.path.join(args.data_dir, 'images-patch2', img_train))
			img = cv2.resize(img, (256, 256))
			cv2.imwrite(os.path.join(args.data_dir, save_dir, "train", "images", img_train), img)
			label = cv2.imread(os.path.join(args.data_dir, 'labels-patch2', img_train))
			label = cv2.resize(label, (256, 256))
			cv2.imwrite(os.path.join(args.data_dir, save_dir, "train", "labels", img_train), label)
		print("Processing Val folder")
		for img_train in list_val:
			img_train = img_train.split('/')
			img_train = img_train[len(img_train)-1]
			img = cv2.resize(img, (256, 256))
			cv2.imwrite(os.path.join(args.data_dir, save_dir, "val", "images", img_train), img)
			label = cv2.imread(os.path.join(args.data_dir, 'labels-patch2', img_train))
			label = cv2.resize(label, (256, 256))
			cv2.imwrite(os.path.join(args.data_dir, save_dir, "val", "labels", img_train), label)
		print("Processing Test folder")
		for img_train in list_test:
			img_train = img_train.split('/')
			img_train = img_train[len(img_train)-1]
			img = cv2.imread(os.path.join(args.data_dir, 'images-patch2', img_train))
			img = cv2.resize(img, (256, 256))
			cv2.imwrite(os.path.join(args.data_dir, save_dir, "test", "images", img_train), img)
			label = cv2.imread(os.path.join(args.data_dir, 'labels-patch2', img_train))
			label = cv2.resize(label, (256, 256))
			cv2.imwrite(os.path.join(args.data_dir, save_dir, "test", "labels", img_train), label)
		print("Finish Processing")
		


if __name__ == "__main__":
	# split_segmentation()
	# split_gan("Total_Photos")
	images2chips.run("Photos_20120715_06")