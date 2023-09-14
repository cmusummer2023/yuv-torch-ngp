import cv2
import imageio
import numpy as np

# Constants
RGB2YUV = np.array([
    [ 0.2126,  0.7152,  0.0722],
    [-0.1146, -0.3854,  0.4991],
    [ 0.5000, -0.4542, -0.0458]
])

##############################################
# READING IN TESTING/VALIDATION IMAGES       #
# Involves conversion from RGB -> __ -> RGB  #
##############################################

"""
1. Reads in an image from img_path
2. Assumes image is from 0.0 - 255.0, normalizes to [0,1]
3. Blends image if needed, 
4. Sets H / W if no values are provided, and return the preprocessed image.

Input image is RGB32, with pixel values in [0.,255.]
Returned image is RGB32, with pixel values in [0., 1.]
"""
def read_and_preprocess_image(img_path, H, W, downscale, blend_a):
    img = imageio.imread(img_path).astype(np.float32) / 255.0

    if H is None or W is None:
        H, W = img.shape[0] // downscale, img.shape[1] // downscale

    # Alpha blending, if necessary
    if img.shape[2] == 4:
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    return img, H, W

"""
read in as regular RGBf32 image
# RGB32->RGB32->RGB32
"""
def read_image_rgb32(img_path, H, W, downscale, blend_a=True):
    return read_and_preprocess_image(img_path, H, W, downscale, blend_a)

"""
1. Uses read_and_preprocess_image to get image that is in range [0., 1.,]
2. Downsamples to uint8
3. Upsample back to float32.

RGB32->RGB8->RGB32 
"""
def read_image_rgb_downsample_rgb8(img_path, H, W, downscale, blend_a=True): 
    img, H, W = read_and_preprocess_image(img_path, H, W, downscale, blend_a)
    
    # Downsample to uint8.
    img = img * 255.0
    img = img.astype('uint8')
    # Upsample back to f32.

    img = img.astype(np.float32) / 255.0 #change back into f32 
    #img = cv2.resize(img, img_wh)
    #img_r = rearrange(img, 'h w c -> (h w) c')

    #img is not rearranged yet; img_r is rearranged 
    return img, H, W

"""
RGB32->YUV420->RGB32
"""
def read_image_rgb_downsample_yuv420(img_path, H, W, downscale, blend_a=True):
    #read in as RGB, convert to YUV, then convert back into RGB 
    img, H, W = read_and_preprocess_image(img_path, H, W, downscale, blend_a)

    # Step 1: YUV colorspace transformation
    # Flattens matrix to a shape of 3 x (n^2)
    rgb_columns = img.transpose(2,0,1).reshape(3,-1)
    # Converts from RGB to YUV colorspace.
    yuv_img = RGB2YUV @ rgb_columns
    # This returns the matrix to dimensions nxnx3,
    yuv_img = yuv_img.reshape(3, H, W).transpose(1,2,0)

    y = yuv_img[:, :, 0]  # Y channel (luma)
    u = yuv_img[:, :, 1]  # U channel (chroma)
    v = yuv_img[:, :, 2]  # V channel (chroma)

    # U / V values is in [-0.5, 0.5]. Convert to [0, 1]
    u += 0.5 
    v += 0.5

    # # Downsample u/v horizontally (definition of 422)
    u = cv2.resize(u, (W//2, H//2), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, (W//2, H//2), interpolation=cv2.INTER_LINEAR)

    # Downsample to uint8
    y *= 255
    u *= 255
    v *= 255

    # Convert u and v to uint8 with clipping and rounding:
    # Convert y to uint8 with rounding
    y = np.round(y).astype(np.uint8)
    u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
    v = np.round(np.clip(v, 0, 255)).astype(np.uint8)

    # At this point, Y, U, V have been computed in uint8. Convert back to RGB.
    v2 = np.zeros(y.shape, dtype=np.uint8)
    v2[0::2, 0::2] = v 
    v2[0::2, 1::2] = v
    v2[1::2, 0::2] = v 
    v2[1::2, 1::2] = v
   
    u2 = np.zeros(y.shape, dtype=np.uint8)
    u2[0::2, 0::2] = u 
    u2[0::2, 1::2] = u
    u2[1::2, 0::2] = u 
    u2[1::2, 1::2] = u

    y = y.astype(np.float32)
    u2 = u2.astype(np.float32)
    v2 = v2.astype(np.float32)
    
    y /= 255.
    u2 /= 255.
    v2 /= 255.

    u2 -= 0.5
    v2 -= 0.5

    ## Conversion back to RGB colorspace
    YUV2RGB = np.linalg.inv(RGB2YUV)

    YUV_2 = np.dstack((y, u2, v2))
    yuv_columns = YUV_2.transpose(2,0,1).reshape(3,-1)
    rgb_img = YUV2RGB @ yuv_columns
    rgb_img = rgb_img.reshape(3, H, W).transpose(1,2,0)

    rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)

    return rgb_img, H, W #return as RGB 

"""
RGB32->YUV420->RGB32
"""
def read_image_rgb_downsample_yuv422(img_path, H, W, downscale, blend_a=True):
    #read in as RGB, convert to YUV, then convert back into RGB 
    img, H, W = read_and_preprocess_image(img_path, H, W, downscale, blend_a)

    # Step 1: YUV colorspace transformation
    # Flattens matrix to a shape of 3 x (n^2)
    rgb_columns = img.transpose(2,0,1).reshape(3,-1)
    # Converts from RGB to YUV colorspace.
    yuv_img = RGB2YUV @ rgb_columns
    # This returns the matrix to dimensions nxnx3,
    yuv_img = yuv_img.reshape(3, H, W).transpose(1,2,0)

    y = yuv_img[:, :, 0]  # Y channel (luma)
    u = yuv_img[:, :, 1]  # U channel (chroma)
    v = yuv_img[:, :, 2]  # V channel (chroma)

    # U / V values is in [-0.5, 0.5]. Convert to [0, 1]
    u += 0.5 
    v += 0.5

    # # Downsample u/v horizontally (definition of 422)
    u = cv2.resize(u, (W//2, H), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, (W//2, H), interpolation=cv2.INTER_LINEAR)

    # Downsample to uint8
    y *= 255
    u *= 255
    v *= 255

    # Convert u and v to uint8 with clipping and rounding:
    # Convert y to uint8 with rounding
    y = np.round(y).astype(np.uint8)
    u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
    v = np.round(np.clip(v, 0, 255)).astype(np.uint8)

    # At this point, Y, U, V have been computed in uint8. Convert back to RGB.
    v2 = np.zeros(y.shape, dtype=np.uint8)
    v2[:, 0::2] = v 
    v2[:, 1::2] = v 
    
    u2 = np.zeros(y.shape, dtype=np.uint8)
    u2[:, 0::2] = u
    u2[:, 1::2] = u

    y = y.astype(np.float32)
    u2 = u2.astype(np.float32)
    v2 = v2.astype(np.float32)
    
    y /= 255.
    u2 /= 255.
    v2 /= 255.

    u2 -= 0.5
    v2 -= 0.5

    ## Conversion back to RGB colorspace
    YUV2RGB = np.linalg.inv(RGB2YUV)

    YUV_2 = np.dstack((y, u2, v2))
    yuv_columns = YUV_2.transpose(2,0,1).reshape(3,-1)
    rgb_img = YUV2RGB @ yuv_columns
    rgb_img = rgb_img.reshape(3, H, W).transpose(1,2,0)

    rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)

    return rgb_img, H, W #return as RGB 

def read_image_rgb_downsample_bggr(img_path, H, W, downscale, blend_a=True):
    #read in as RGB, convert to YUV, then convert back into RGB 
    img, H, W = read_and_preprocess_image(img_path, H, W, downscale, blend_a)

    (height, width) = img.shape[:2]
    (R,G,B) = cv2.split(img)

    bayer = np.empty((height, width), np.uint8)
        
    #bggr?
    bayer[0::2, 0::2] = B[0::2, 0::2] # top left
    bayer[0::2, 1::2] = G[0::2, 1::2] # top right
    bayer[1::2, 0::2] = G[1::2, 0::2] # bottom left
    bayer[1::2, 1::2] = R[1::2, 1::2] # bottom right
    img = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)

    img = img.astype(np.float32) / 255.0

    #img is not rearranged 
    return img, H, W

##############################
# READING IN TRAINING IMAGES #
# we change how the images are being stored here 
##############################

#read and STORE as RGB 8 bit image from RGB32->RGB8->RGB32 images
def read_image_rgb8(img):
    '''
    img = imageio.imread(img_path).astype(np.float32)
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] / 255.0 * img[..., -1:] + (255 - img[..., -1:])
        else:
            img = img[..., :3] / 255.0 * img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = img.astype('uint8') #turn back into 8-bit format 

    img = rearrange(img, 'h w c -> (h w) c')
    '''
    #img, H, W = read_image_rgb_downsample_rgb8(img_path, H, W, downscale) #(800, 800, 3) shape 
    img = img * 255.0 
    img = img.astype('uint8')
    return img

# read and STORE as YUV420 8-bit image from RGB32->YUV420 8->RGB32 images 
def read_image_yuv420_8(img):
    '''
    img = imageio.imread(img_path).astype(np.float32) 
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] / 255.0 * img[..., -1:]  + (255 - img[..., -1:])
        else:
            img = img[..., :3] / 255.0 * img[..., -1:]
    #results are a 3-channel image
    img = cv2.resize(img, img_wh)
    '''
    #img, H, W = read_image_rgb_downsample_yuv420(img_path, H, W, downscale)
    img = img * 255.0 
    img = img.astype('uint8')
    #i had to had to changed the inputs to the cvtColor function back into integers but i wasn't sure if that would change the rounding
    # error from cvtColor function (https://stackoverflow.com/questions/55128386/python-opencv-depth-of-image-unsupported-cv-64f) 
    # OR i could have replace the PEMDAS (changing dividing 255 right after reading to dividing at the end) but not sure how to do without messing thigns up 
    #might do later     

    #Converting to YUV 4:2:0 format (not sure if needs to be within 255 or not? )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420) #two dim with 1 channel (1200 x 800)

    # originally after the cv2.resize method for regular rgb images (i think for easier indexing when iterating thru dataset)
    #img = rearrange(img, 'h w c -> (h w) c') #two dimensional w/ 3 channels (160000 x 3)
    return img

#only applicable to train datasets 
# read and STORE as YUV420 f32 image  
def read_image_yuv420_f32(img):
    img_width, img_height = img.shape[1], img.shape[0]

    # Step 1: YUV colorspace transformation
    # Flattens matrix to a shape of 3 x (n^2)
    rgb_columns = img.transpose(2,0,1).reshape(3,-1)
    # Converts from RGB to YUV colorspace.
    yuv_img = RGB2YUV @ rgb_columns
    # This returns the matrix to dimensions nxnx3,
    yuv_img = yuv_img.reshape(3, img_height, img_width).transpose(1,2,0)

    y = yuv_img[:, :, 0]  # Y channel (luma)
    u = yuv_img[:, :, 1]  # U channel (chroma)
    v = yuv_img[:, :, 2]  # V channel (chroma)


    # # Downsample u/v horizontally AND vertically (definition of 420)
    u = cv2.resize(u, (img_width//2, img_height//2), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, (img_width//2, img_height//2), interpolation=cv2.INTER_LINEAR)

    # We concatenate it (stack it horizontally), to get one matrix # n/2 x n/2
    uv = np.concatenate((u, v), axis=1) # n/2 x n/2.
    
    # Merge y and u/v channels into one array.
    # n*2 x n.
    yuv420 = np.concatenate((y, uv), axis=0)

    yuv420 = yuv420.astype(np.float32)

    return yuv420

"""
Takes the preprocessed image that has gone through (img -> yuv422 -> img)
and converts it to YUV422.

Input: rgb_img in range [0.,1.]
Output: yuv422 array with the following configuration:
 ________
|       |
|   Y   |
|       |
|_______|
| U | V |
|   |   |
|   |   |
|___|___|
"""
def read_image_yuv422_8(img):    
    # Vectorizing YUV colorspace transformation    
    img_width, img_height = img.shape[1], img.shape[0]

    assert(np.max(img) <= 1.)

    # This flattens the matrix out to a shape of 3 x (n^2)
    rgb_columns = img.transpose(2,0,1).reshape(3,-1)
    # Converts from RGB to YUV colorspace.
    yuv_img = RGB2YUV @ rgb_columns
    # This returns the matrix to dimensions nxnx3,
    yuv_img = yuv_img.reshape(3, img_height, img_width).transpose(1,2,0)

    y = yuv_img[:, :, 0]  # Y channel (luma)
    u = yuv_img[:, :, 1]  # U channel (chroma)
    v = yuv_img[:, :, 2]  # V channel (chroma)

    u += 0.5
    v += 0.5

    # # Downsample u/v horizontally
    u = cv2.resize(u, (img_width//2, img_height), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, (img_width//2, img_height), interpolation=cv2.INTER_LINEAR)

    y *= 255
    u *= 255
    v *= 255

    # Convert u and v to uint8 with clipping and rounding:
    # Convert y to uint8 with rounding
    y = np.round(y).astype(np.uint8)
    u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
    v = np.round(np.clip(v, 0, 255)).astype(np.uint8)

    # We concatenate it (stack it horizontally), to get one matrix # n x n
    uv = np.concatenate((u, v), axis=1) # n x n.
    
    # Merge y and u/v channels into one array.
    # n*2 x n.
    yuv422 = np.concatenate((y, uv), axis=0)    

    return yuv422

#only applicable to training datasets 
# read and STORE as YUV 422 f32 image  
def read_image_yuv422_f32(img):
    # Vectorizing YUV colorspace transformation    
    # This flattens the matrix out to a shape of 3 x (n^2)
    img_width, img_height = img.shape[1], img.shape[0]

    rgb_columns = img.transpose(2,0,1).reshape(3,-1)
    # Converts from RGB to YUV colorspace.
    yuv_img = RGB2YUV @ rgb_columns
    # This returns the matrix to dimensions nxnx3,
    yuv_img = yuv_img.reshape(3, img_height, img_width).transpose(1,2,0)

    y = yuv_img[:, :, 0]  # Y channel (luma)
    u = yuv_img[:, :, 1]  # U channel (chroma)
    v = yuv_img[:, :, 2]  # V channel (chroma)

    # Downsample u/v horizontally
    # Given an RGB image of dimensions nxnx3, u will be n x n/2 x 1, and v will be n x n/2 x 1
    u = cv2.resize(u, (img_width//2, img_height), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, (img_width//2, img_height), interpolation=cv2.INTER_LINEAR)

    # We concatenate it (stack it horizontally), to get one matrix # n x n
    uv = np.concatenate((u, v), axis=1) # n x n.
    
    # Merge y and u/v channels into one array.
    # n*2 x n.
    yuv422 = np.concatenate((y, uv), axis=0)

    yuv422 = yuv422.astype(np.float32)

    return yuv422

def read_image_bggr_8(img): 
    (height, width) = img.shape[:2]
    (R, G, B) = cv2.split(img)

    img = img * 255.0

    bayer = np.empty((height, width), np.uint8)
        
    #bggr?
    bayer[0::2, 0::2] = B[0::2, 0::2] # top left
    bayer[0::2, 1::2] = G[0::2, 1::2] # top right
    bayer[1::2, 0::2] = G[1::2, 0::2] # bottom left
    bayer[1::2, 1::2] = R[1::2, 1::2] # bottom right
    img = bayer.astype(np.uint8)

    return img  #(800, 800)