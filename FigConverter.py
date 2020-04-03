import argparse
import os
import subprocess


def convert_png_jpg(fullpath):
    newpath = fullpath[:-3] + 'jpg'
    subprocess.run(['convert', fullpath, newpath])
    os.remove(fullpath)


def convert_dir(dir):
    if os.path.isdir(dir):
        subdirs = os.listdir(dir)
        for sd in subdirs:
            convert_dir(dir + '/' + sd)
    elif dir[-3:].lower() == 'png':
        convert_png_jpg(dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir', default='/home/thanhtung/Dropbox/CatastrophicGAN/AISTATS2020-Tung/figs/')

    args = parser.parse_args()
    convert_dir(args.indir)
