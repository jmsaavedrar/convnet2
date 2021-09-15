"""
@author: jsaavedr
"""
import numpy as np
import skimage.transform as transf
import argparse
import skimage.io as io
import skimage.color as color
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import skimage.morphology as morph
import os

#to uint8
def read_image_test(filename, number_of_channels):
    """ read_image using skimage
        The output is a 3-dim image [H, W, C]
    """        
    if number_of_channels  == 1 :            
        image = io.imread(filename, as_gray = True)
        image = toUINT8(image)
        assert(len(image.shape) == 2)
        image = np.expand_dims(image, axis = 2) #H,W,C                    
        assert(len(image.shape) == 3 and image.shape[2] == 1)
    elif number_of_channels == 3 :        
        image = io.imread(filename)
        if(len(image.shape) == 2) :
            image = color.gray2rgb(image)
        elif image.shape[2] == 4 :
            image = color.rgba2rgb(image)     
        image = toUINT8(image)        
        assert(len(image.shape) == 3 and image.shape[2]==3)
    else:
        raise ValueError("number_of_channels must be 1 or 3")
    if not os.path.exists(filename):
        raise ValueError(filename + " does not exist!")
    return image

def toUINT8(image) :
    if image.dtype == np.float64 :
        image = image * 255
    elif image.dtype == np.uint16 :
        image = image >> 8        
    image[image<0]=0
    image[image>255]=255
    image = image.astype(np.uint8, copy=False)
    return image

def resize_image(image, imsize):
    """
    imsize = (h,w)
    """ 
    image_out = transf.resize(image, imsize)    
    image_out = toUINT8(image_out)
    return image_out
   
def resize_image_keeping_aspect(image, output_size):
    """
    this process resizes the input image keeping the aspect ratio
    max_size : maximum size 
    """
    cur_height = image.shape[0]
    cur_width = image.shape[1]    
    
    factor_y = output_size[0] / cur_height
    factor_x = output_size[1] / cur_width
    factor = np.min([factor_y, factor_x])
    target_height = int(factor * cur_height)
    target_width = int(factor * cur_width)
    if len(image.shape) == 2 :    
        output_shape = output_size
    elif len(image.shape) == 3 :
        output_shape = (output_size[0], output_size[1], image.shape[2])
    else :
        raise ValueError("imgproc: input image format is incorrect!")
    image_resized = toUINT8(transf.resize(image, [target_height, target_width]))
    image_out = np.zeros(output_shape, np.uint8) + 255
    center_y = int(output_size[0] / 2)
    center_x = int(output_size[1] / 2)
    #lt: left_top corner
    y_lt = center_y - int(target_height / 2)
    x_lt = center_x - int(target_width / 2)
    image_out[y_lt : y_lt + target_height, x_lt : x_lt + target_width] = image_resized;
    #image_out = toUINT8(image_out)
    return image_out

def image_crop_rgb(image, bg_color, padding = 0):
    assert(len(image.shape) == 3 and image.shape[2] == 3)
    assert(len(bg_color) == 3)
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]    
    b_red = (red == bg_color[0])
    b_green = (green == bg_color[1])
    b_blue = (blue == bg_color[2])
    _bin = b_red & b_green & b_blue
    _bin = np.bitwise_not(_bin)    
    row_proyection = np.sum(_bin, 1)
    col_proyection = np.sum(_bin, 0)
    xs_pos = np.where(col_proyection > 0)[0]
    ys_pos = np.where(row_proyection > 0)[0]
    if (len(xs_pos) > 1 and len(ys_pos > 0)) :    
        x_min = xs_pos[0]
        x_max = xs_pos[-1]
        y_min = ys_pos[0]
        y_max = ys_pos[-1]
        cropped_image = image[y_min:y_max, x_min:x_max]
        if padding > 0 :
            im_h = cropped_image.shape[0] + 2*padding
            im_w = cropped_image.shape[1] + 2*padding
            shape = (im_h, im_w, 3)
            new_image = np.ones(shape, np.uint8)*np.array(bg_color);
            new_image = new_image.astype(np.uint8)
            new_image[padding : padding + cropped_image.shape[0], padding : padding + cropped_image.shape[1], :] = cropped_image;
        else :    
            new_image = cropped_image
    else :
        new_image = image
    return new_image

def image_crop_gray(image, bg_color, padding = 0):
    assert(len(image.shape) == 3 and image.shape[2] == 1)        
    mask = (image[:,:,0] != bg_color)        
    row_proyection = np.sum(mask, 1)
    col_proyection = np.sum(mask, 0)
    xs_pos = np.where(col_proyection > 0)[0]
    ys_pos = np.where(row_proyection > 0)[0]
    if (len(xs_pos) > 1 and len(ys_pos > 0)) :    
        x_min = xs_pos[0]
        x_max = xs_pos[-1]
        y_min = ys_pos[0]
        y_max = ys_pos[-1]
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        if padding > 0 :
            im_h = cropped_image.shape[0] + 2*padding
            im_w = cropped_image.shape[1] + 2*padding
            shape = (im_h, im_w, 1)
            new_image = np.ones(shape, np.uint8)*bg_color;
            new_image = new_image.astype(np.uint8)            
            new_image[padding : padding + cropped_image.shape[0], padding : padding + cropped_image.shape[1], :] = cropped_image;
        else :    
            new_image = cropped_image
    else :
        new_image = image
    return new_image
     
def process_sketch(image, output_size):
    new_image = image_crop_rgb(image, (255,255,255), padding = 20)
    new_image = resize_image_keeping_aspect(new_image, output_size)
    one_channel = new_image[:,:,0]
    one_channel = morph.erosion(one_channel, morph.square(3))    
    new_image[:,:,0] = one_channel;
    new_image[:,:,1] = one_channel;
    new_image[:,:,2] = one_channel;         
    return new_image

def process_mnist(image, output_size):
    new_image = image_crop_gray(image, 255, padding = 10)
    new_image = resize_image_keeping_aspect(new_image, output_size)             
    return new_image

def process_image(image, output_size):
    image = image_crop_rgb(image, (255,255,255), padding = 20)
    image = resize_image_keeping_aspect(image, output_size)
    return image    

def process_image_visual_attribute(image, output_size):
    image = image_crop_rgb(image, (255,255,255), padding = 0)
    image = resize_image_keeping_aspect(image, output_size)
    return image

def create_processing_function(imgproc_params):
    keep_aspect_ratio = imgproc_params['keep_aspect_ratio']
    padding_value = imgproc_params['padding_value']
    with_crop = imgproc_params['with_crop']
    bg_color = imgproc_params['bg_color']
    n_channels = imgproc_params['n_channels']
    
    def process_fun(image, output_size):
        if with_crop :
            if n_channels == 1 :
                f_crop = image_crop_gray
            if n_channels == 3 :
                f_crop = image_crop_rgb
            image =  f_crop(image, bg_color, padding_value)
        if keep_aspect_ratio :
            f_resize = resize_image_keeping_aspect
        else :
            f_resize = resize_image
        image = f_resize(image, output_size)
        return image
    return process_fun       
    
def change_color(image):
    """
    change the color of an image
    """    
    hsv = color.rgb2hsv(image)
    r1 = int(random.uniform(0.4,0.9)*10)/10
    r2 = int(random.uniform(0.4,0.9)*10)/10
    hsv[:,:,0] = hsv[:,:,0]*r1
    hsv[:,:,1] = hsv[:,:,1]*r1
    print('{} {}'.format(r1,r2))
    
    return color.hsv2rgb(hsv)
    
"""
unit test
"""
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Unit test for image processsing")
    parser.add_argument("-image", type = str, help = "input image filename", required = True )
    
    #parser.add_argument("-max_size", type = int, help = "max size of the image dimensions", required = True )
    pargs = parser.parse_args()     
    filename = pargs.image
    
    imgproc_params={}
    imgproc_params['keep_aspect_ratio'] = True
    imgproc_params['padding_value'] = 0
    imgproc_params['with_crop'] = True
    #a =0    
    #imgproc_params['bg_color'] = [int(v.strip()) for v in a.split()]
    imgproc_params['bg_color'] = 0
    imgproc_params['n_channels'] = 1    
    #image = process_sketch(image, (224,224))
    #image = process_mnist(image, (50,50))
    image = read_image_test(filename, imgproc_params['n_channels'])        
    process_fun = create_processing_function(imgproc_params)
    image = process_fun(image, (31,31))
    print(image.shape)
    plt.imshow(image)
    plt.show()
    #print(image.shape)   
    #new_image = tf.image.adjust_hue(image, 0.5)
    #new_image = tf.image.adjust_contrast(image, 2)
    #new_image = tf.image.adjust_brightness(image, -0.3)
#     height = image.shape[0]
#     width = image.shape[1]
#     prob = tf.random.uniform((),0,1)
#     if prob > 0.5:
#         #new_image = tf.image.flip_left_right(image)
#         #new_image = tf.image.adjust_brightness(image, 0.2)#random.uniform(-0.5,0.5))
#         #new_image = tf.image.adjust_saturation(image,0.6)
# #         a = tf.random.uniform((),0.5,0.9)
# #         print("{} {}".format(prob, a))
#         new_image = tf.image.central_crop(image, central_fraction = 0.8)
#         new_image = tf.cast(tf.image.resize(new_image, (height, width)), tf.uint8)
#     else :
#         new_image = image
#     #print(new_image)            
#     fig, xs = plt.subplots(1,2)
#     xs[0].imshow(image)
#     xs[1].imshow(new_image)
#     plt.show()
    """max_size = pargs.max_size
    print(filename)
    try:
        image = io.imread(filename)
        image_out = process_image_keeping_aspect(image, (max_size, max_size))
        fig, xs = plt.subplots(1,2)
        xs[0].imshow(image)
        xs[1].imshow(image_out)
        plt.show()
    except ValueError :
        print("Error reading {}".format(filename))
    
    """
    