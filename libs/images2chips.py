import cv2
import os
import numpy as np

# from libs.config import LABELMAP_DEER, INV_LABELMAP_DEER

width = 320
height = 320
stride = 320

LABELMAP_DEER = {
    0 : (0, 0, 128),
    1 : (0, 0, 0),
    2 : (0, 128, 0),
    3 : (0,  128,  128),
}

# Color (BGR) to class
INV_LABELMAP_DEER = {
    (0,  0, 128) : 0,
    (0,  0, 0) : 1,
    (0, 128, 0) : 2,
    (0, 128, 128) : 3,
}

def color2class(orthochip, img):
    ret = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    ret = np.dstack([ret, ret, ret])
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

    # Skip any chips that would contain magenta (IGNORE) pixels
    seen_colors = set( [tuple(color) for color in colors] )
    # IGNORE_COLOR = LABELMAP[0]
    # if IGNORE_COLOR in seen_colors:
    #     return None, None

    for color in colors:
        print(color)
        locs = np.where( (img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2]) )
        if tuple(color) in INV_LABELMAP_DEER:
            ret[ locs[0], locs[1], : ] = INV_LABELMAP_DEER[ tuple(color) ]
        else:
            ret[ locs[0], locs[1], : ] = 3

    return orthochip, ret

def image2tile(prefix, scene, orthofile, labelfile, windowx=width, windowy=height, stridex=stride, stridey=stride):

    ortho = cv2.imread(orthofile)
    # label = cv2.imread(labelfile)

    # Not using elevation in the sample - but useful to incorporate it ;)
    # eleva = cv2.imread(elevafile, -1)

    # assert(ortho.shape[0] == label.shape[0])
    # assert(ortho.shape[1] == label.shape[1])

    shape = ortho.shape

    xsize = shape[1]
    ysize = shape[0]
    print(f"converting image {orthofile} {xsize}x{ysize} to patches size {windowx} ...")

    counter = 0

    for xi in range(0, shape[1] - windowx, stridex):
        for yi in range(0, shape[0] - windowy, stridey):

            orthochip = ortho[yi:yi+windowy, xi:xi+windowx, :]
            # labelchip = label[yi:yi+windowy, xi:xi+windowx, :]


            orthochip_filename = os.path.join(prefix + '-images-patch2', scene + '-extra-' + str(windowx) + '-' + str(counter).zfill(6) + '.png')
            # labelchip_filename = os.path.join(prefix, 'labels-patch2', scene + '-' + str(windowx) + '-' + str(counter).zfill(6) + '.png')

            cv2.imwrite(orthochip_filename, orthochip)
            # cv2.imwrite(labelchip_filename, labelchip)
            counter += 1


def run(prefix):

    image_dir = prefix + '-images-patch2'
    label_dir = 'labels-patch2'
    if not os.path.exists(image_dir):
        # os.mkdir(os.path.join(prefix, image_dir))
        os.mkdir(image_dir)

    # if not os.path.exists( os.path.join(prefix, label_dir) ):
    #     os.mkdir(os.path.join(prefix, label_dir))
    
    print("Start processing")
    # list_images = os.listdir(os.path.join(prefix, 'JPEGImages'))
    list_images = os.listdir(prefix)
    for image in list_images:
        image_file = os.path.join(prefix, image)
        label_file = ''
        # label_file = os.path.join(prefix, 'SegmentationClassPNG', image)
        for i in range(800, 2500, 300):
            if os.path.exists(image_file):
                image2tile(prefix, image.split('.')[0], image_file, label_file, windowx=i + 100, windowy=i, stridex=300, stridey=300)
    print("End processing")


    # lines = [ line for line in open(f'{prefix}/index.csv') ]
    # num_images = len(lines) - 1
    # print(f"converting {num_images} images to chips - this may take a few minutes but only needs to be done once.")
    # train_file = open(os.path.join(prefix,'train.txt'),'r')
    # valid_file = open(os.path.join(prefix,'valid.txt'),'r')
    # test_file = open(os.path.join(prefix,'test.txt'),'r')
    # # lines = train_file.readlines() + valid_file.readlines()
    # lines = test_file.readlines()

    # for line in os.listdir(os.path.join(prefix, "JPEGImages")):
    # for line in lines:
    #     line = line.replace("\n","")
    #     orthofile = os.path.join(prefix, 'JPEGImages', line)
    #     labelfile = os.path.join(prefix, 'SegmentationClassPNG', line)
    #     if os.path.exists(orthofile) and os.path.exists(labelfile):
    #         image2tile(prefix, line.split('.')[0], orthofile, labelfile)

if __name__ == "__main__":
    run("Photos")

