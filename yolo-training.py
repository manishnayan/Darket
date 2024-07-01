
"""
    Purpose             : To automatically train YOLO model.
    Device Requirment   : Python Interpreter + Darknet Environment setup + GPUs
    Author              : Manish Arya
    Last Edited         : 01 July 2024

"""
# !/usr/bin/env python
import os
import glob
import math
import time
import tqdm
import shutil
import random
import argparse
import subprocess

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection Training")
    parser.add_argument("--EnvDir", type=str, default="",
                        help="It is path for the environment folder having a folder named train (consisting all the data), classes.txt file and .cfg file")
    parser.add_argument("--init_weight", type=str, default="",
                        help="It is path for the environment folder having a folder named train (consisting all the data), classes.txt file and .cfg file")
    parser.add_argument("--split", type=float, default=0.1,
                        help="Percentage to split data for validation set.")
    parser.add_argument("--gpus", type=str, default="0",
                        help="In case of multiple gpus define gpu on which you want to run yolo training like: (for two gpus: '--gpus 0,1', to use one of them: '--gpus 0 or --gpus 1')")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.split < 1, "Split should be a float between zero and one (non-inclusive)"
    if args.EnvDir and not os.path.exists(args.EnvDir):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.EnvDir))))
    if args.init_weight and not os.path.exists(args.init_weight):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.init_weight))))

def load_files(EnvDir):

    cfg_file = glob.glob(os.path.join(EnvDir, "*.cfg"))[0]
    classes_file = os.path.join(EnvDir, 'classes.txt')
    train_dir = os.path.join(EnvDir, 'train')
    val_dir = os.path.join(EnvDir, 'validation')

    if not os.path.exists(cfg_file):
        raise(ValueError("'.cfg' file missing at: {}".format(os.path.abspath(EnvDir))))
    if not os.path.exists(classes_file):
        raise(ValueError("'classes.txt' file missing at: {}".format(os.path.abspath(EnvDir))))
    if not os.path.exists(train_dir):
        raise(ValueError("'train' folder missing at: {}".format(os.path.abspath(EnvDir))))
    if not os.path.exists(val_dir):
        raise(ValueError("'validation' folder missing at: {}".format(os.path.abspath(EnvDir))))
    os.makedirs(os.path.join(EnvDir, 'backup'), exist_ok=True)

    return cfg_file, classes_file, train_dir, val_dir

def data_split(train_dir, val_dir, split):

    if not os.path.exists(val_dir):
        print('\nSplitting data is in process...')
        time.sleep(2.4)
        
        os.makedirs(val_dir, exist_ok=True)
        images = glob.glob(
                os.path.join(train_dir, "*.jpg")) + \
                glob.glob(os.path.join(train_dir, "*.png")) + \
                glob.glob(os.path.join(train_dir, "*.jpeg"))

        random.shuffle(images)
        val_images = random.sample(images, math.ceil(len(images)*split))

        for imgSrc in tqdm.tqdm(val_images):

            imgName = imgSrc.split('/')[-1]
            ext = imgName.split('.')[-1]
            imgDst = os.path.join(val_dir, imgName)
            
            txt_src_split = imgSrc.split('.')[:-1]
            txt_src_split.append('txt')
            txtSrc = ".".join(txt_src_split)
            txtDst = imgDst.replace(ext, 'txt')
            try:
                shutil.move(txtSrc, txtDst)
                shutil.move(imgSrc, imgDst)
            except:
                continue
        print('Done !!')  
        time.sleep(2.4)  
    else:
        print("\nValidation folder already exist_______SKIPPING Data Split !!")
        time.sleep(2.4)

def generate_train_val_list(EnvDir, split):

    val_dir = os.path.join(EnvDir, 'validation')
    train_dir = os.path.join(EnvDir, 'train')
    data_split(train_dir, val_dir, split)
    DIRs = [train_dir, val_dir]

    print('\nGenerating train.txt and validation.txt file...')
    time.sleep(2.4)
    for Dir in tqdm.tqdm(DIRs):
        DirName = Dir.split('/')[-1]
        images = glob.glob(
                os.path.join(Dir, "*.jpg")) + \
                glob.glob(os.path.join(Dir, "*.png")) + \
                glob.glob(os.path.join(Dir, "*.jpeg"))
        with open(os.path.join(EnvDir,DirName+".txt"), "w") as outfile:
            for image in images:
                outfile.write(image)
                outfile.write("\n")
            outfile.close()
    print('Done !!')
    time.sleep(2.4)

def trainModel(args):

    classes = list()
    print('\nLoading Files...')
    time.sleep(2.4)
    cfg_file, classes_file, train_dir, val_dir = load_files(args.EnvDir)
    data_file = os.path.join(args.EnvDir, 'obj.data')

    with open(classes_file, "r") as f:
        for line in f:
            classes.append(line.strip())
        f.close()

    DATA = [f"classes = {len(classes)}",
            f"train  = {'.'.join([train_dir, 'txt'])}",
            f"valid  = {'.'.join([val_dir, 'txt'])}",
            f"names = {classes_file}",
            f"backup = {os.path.join(args.EnvDir, 'backup')}"]
    with open(data_file, "w") as outfile:
        for data in DATA:
            outfile.write(data)
            outfile.write("\n")
        outfile.close()
    print('Done !!')
    time.sleep(2.4)

    if args.init_weight:
        print(f'\nTraining started with pre-defined weights: {args.init_weight}\n') 
        time.sleep(2.4)
        subprocess.run(f"./darknet detector train {data_file} {cfg_file} {args.init_weight} -dont_show -map -gpus {args.gpus}", shell=True, check=True)
    else:
        print(f'\nTraining started with Random initial weights.\n') 
        time.sleep(2.4)
        subprocess.run(f"./darknet detector train {data_file} {cfg_file} -dont_show -map -gpus {args.gpus}", shell=True, check=True)

    print('\nTraining Done !!')
    time.sleep(2.4)
    
    
def main():

    args = parser()
    check_arguments_errors(args)
    generate_train_val_list(args.EnvDir, args.split)
    trainModel(args)


if __name__ == "__main__":
    # Calling main method !!
    main()
