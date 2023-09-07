import cv2
import imageio
import numpy as np
#from einops import rearrange


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, ((img + 0.055) / 1.055)**2.4, img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img > limit, 1.055 * img**(1 / 2.4) - 0.055, 12.92 * img)
    img[img > 1] = 1  # "clamp" tonemapper
    return img
    

def read_image(f_path, H, W, downscale):
    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
    if H is None or W is None:
        H = image.shape[0] // downscale
        W = image.shape[1] // downscale

    # add support for the alpha channel as a mask.
    if image.shape[-1] == 3: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    if image.shape[0] != H or image.shape[1] != W:
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        
    image = image.astype(np.float32) / 255 # [H, W, 3/4] 
    # output 32-bit, [0,1] range
    # eventually RGB in a numpy array... but no rearranging 
    # not sure how they change the BACKGROUND? they KEEP the alpha background?? 

    return image, H, W

 

#read in as regular RGBf32 image
# RGB32->RGB32->RGB32 
def read_image_rgb32(img_path, H, W, downscale, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32) / 255.0

    if H is None or W is None:
        H = img.shape[0] // downscale
        W = img.shape[1] // downscale

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    return img, H, W


# READING IN TESTING/VALIDATION IMAGES
# involves conversion from RGB -> __ -> RGB

# changing RGB32->RGB8->RGB32
def read_image_rgb_downsample_rgb8(img_path, H, W, downscale, blend_a=True): 
    img = imageio.imread(img_path).astype(np.float32) / 255.0

    if H is None or W is None:
        H = img.shape[0] // downscale
        W = img.shape[1] // downscale

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    img = img * 255.0 # downsample to int
    img = img.astype('uint8')

    img = img.astype(np.float32) / 255.0 #change back into f32 
    #img = cv2.resize(img, img_wh)
    #img_r = rearrange(img, 'h w c -> (h w) c')

    #img is not rearranged yet; img_r is rearranged 
    return img, H, W



#Only applicable to test and validation datasets 
#changing RGB32->YUV420->RGB32
def read_image_rgb_downsample_yuv420(img_path, H, W, downscale, blend_a=True):
    #read in as RGB, convert to YUV, then convert back into RGB 
    img = imageio.imread(img_path).astype(np.float32)
    if H is None or W is None:
        H = img.shape[0] // downscale
        W = img.shape[1] // downscale

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] / 255.0 * img[..., -1:] + (255 - img[..., -1:])
        else:
            img = img[..., :3] / 255.0 * img[..., -1:]

    img = img.astype('uint8')

    #turn into RGB->YUV 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
    #turn into YUV->RGB
    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB_I420)

    img = img.astype(np.float32) / 255.0

    #img is not rearranged 
    return img, H, W


#Only applicable to test and validation datasets 
#changing RGB->YUV422->RGB
def read_image_rgb_downsample_yuv422(img_path, H, W, downscale, blend_a=True):
    #read in as RGB, convert to YUV, then convert back into RGB 
    _, y, u, v, H, W = read_image_yuv422_f32_helper(img_path, H, W, downscale, blend_a=True)
    #NOTE that y, u, v are int type  

    #print(v.shape)
    #no interpolation 
    v2 = np.zeros(y.shape, dtype=np.uint8)
    v2[:, 0::2] = v 
    v2[:, 1::2] = v 
    
    u2 = np.zeros(y.shape, dtype=np.uint8)
    u2[:, 0::2] = u
    u2[:, 1::2] = u

    c = y.astype(np.long) - 16
    d = u2.astype(np.long) - 128
    e = v2.astype(np.long) - 128

    r = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
    g = np.clip(( 298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
    b = np.clip(( 298 * c + 516 * d + 128) >> 8, 0, 255)

    img = cv2.merge((r,g,b))
    #print(img.shape) #ideally (800, 800, 3)

    '''
    y0 = np.expand_dims(y[::, ::2], axis=2)
    u = np.expand_dims(u, axis=2)
    y1 = np.expand_dims(y[::, 1::2], axis=2)
    v = np.expand_dims(v, axis=2)

    img_yuyv = np.concatenate((y0, u, y1, v), axis=2)
    img_yuyv_cvt = img_yuyv.reshape(img_yuyv.shape[0], img_yuyv.shape[1] * 2, int(img_yuyv.shape[2] / 2))
    
    #now we convert back into integers for the conversion 
    img_rgb_restored = cv2.cvtColor(img_yuyv_cvt, cv2.COLOR_YUV2RGB_YUYV)
    '''
    img = img.astype(np.float32)/255.0

    return img, H, W #return as RGB 



#only applicable to training datasets 
# read and STORE as YUV 422 f32 image FROM ORIGINAL PNG IMAGE 
def read_image_yuv422_f32_helper(img_path, H, W, downscale, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32) 
    if H is None or W is None:
        H = image.shape[0] // downscale
        W = image.shape[1] // downscale

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] / 255.0 * img[..., -1:]  + (255 - img[..., -1:])
        else:
            img = img[..., :3] / 255.0 * img[..., -1:]
    #results are a 3-channel image

    r, g, b = cv2.split(img)
    rows, cols = r.shape

    # Compute Y, U, V according to the formula described here:
    # https://developer.apple.com/documentation/accelerate/conversion/understanding_ypcbcr_image_formats
    # U applies Cb, and V applies Cr

    # Use BT.709 standard "full range" conversion formula
    y = 0.2126*r + 0.7152*g + 0.0722*b
    u = 0.5389*(b-y) + 128
    v = 0.6350*(r-y) + 128

    # Downsample u horizontally
    u = cv2.resize(u, (cols//2, rows), interpolation=cv2.INTER_LINEAR)

    # Downsample v horizontally
    v = cv2.resize(v, (cols//2, rows), interpolation=cv2.INTER_LINEAR)

    # Convert y to uint8 with rounding
    y = np.round(y).astype(np.uint8)

    # Convert u and v to uint8 with clipping and rounding: 
    u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
    v = np.round(np.clip(v, 0, 255)).astype(np.uint8)

    # Interleave u and v:
    uv = np.zeros_like(y)
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    # Merge y and uv channels
    yuv422 = cv2.merge((y, uv))
    #should have shape (800, 800, 2)
    
    img = yuv422.astype(np.float32) / 255.0
    #dividing by 255 here 

    # originally after the cv2.resize method for regular rgb images (i think for easier indexing when iterating thru dataset)
    
    #note that y, u, v are NOT normalized (in range [0, 255]) while img is normalized 
    return img, y, u, v, H, W


def read_image_rgb_downsample_bggr(img_path, H, W, downscale, blend_a=True):
    #read in as RGB, convert to YUV, then convert back into RGB 
    img = imageio.imread(img_path).astype(np.float32)
    if H is None or W is None:
        H = img.shape[0] // downscale
        W = img.shape[1] // downscale

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] / 255.0 * img[..., -1:] + (255 - img[..., -1:])
        else:
            img = img[..., :3] / 255.0 * img[..., -1:]

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






# READING IN TRAINING IMAGES 
# we change how the images are being stored here 

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
    img = img * 255.0
    img = img.astype('uint8')
    #i had to had to changed the inputs to the cvtColor function back into integers but i wasn't sure if that would change the rounding
    # error from cvtColor function (https://stackoverflow.com/questions/55128386/python-opencv-depth-of-image-unsupported-cv-64f) 
    # OR i could have replace the PEMDAS (changing dividing 255 right after reading to dividing at the end) but not sure how to do without messing thigns up 
    #might do later     
    
    #Converting to YUV 4:2:0 format (not sure if needs to be within 255 or not? )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420) #two dim with 1 channel (1200 x 800)

    img = img.astype(np.float32) / 255.0
    #dividing by 255 here 

    # originally after the cv2.resize method for regular rgb images (i think for easier indexing when iterating thru dataset)
    #img = rearrange(img, 'h w c -> (h w) c') #two dimensional w/ 3 channels (160000 x 3)

    return img



# read and STORE as YUV 422 8 bit image from RGB32->YUV422 8->RGB32
# FROM ORIGINAL IMAGE?? 
def read_image_yuv422_8(img):
    
    #results are a 3-channel image
    #img = cv2.resize(img, img_wh)
    
    #from downsampled iamge??
    #img, H, W = read_image_rgb_downsample_yuv422(img_path, H, W, downscale) 
    
    img = img * 255.0 #float within [0, 255] range 

    r, g, b = cv2.split(img)
    rows, cols = r.shape

    # Compute Y, U, V according to the formula described here:
    # https://developer.apple.com/documentation/accelerate/conversion/understanding_ypcbcr_image_formats
    # U applies Cb, and V applies Cr

    # Use BT.709 standard "full range" conversion formula
    y = 0.2126*r + 0.7152*g + 0.0722*b
    u = 0.5389*(b-y) + 128
    v = 0.6350*(r-y) + 128

    # Downsample u horizontally
    u = cv2.resize(u, (cols//2, rows), interpolation=cv2.INTER_LINEAR)

    # Downsample v horizontally
    v = cv2.resize(v, (cols//2, rows), interpolation=cv2.INTER_LINEAR)

    # Convert y to uint8 with rounding
    y = np.round(y).astype(np.uint8)

    # Convert u and v to uint8 with clipping and rounding: 
    u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
    v = np.round(np.clip(v, 0, 255)).astype(np.uint8)

    # Interleave u and v:
    uv = np.zeros_like(y)
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    # Merge y and uv channels
    yuv422 = cv2.merge((y, uv))
    #should have shape (800, 800, 2)
    
    # originally after the cv2.resize method for regular rgb images (i think for easier indexing when iterating thru dataset)
    #note that y, u, v are NOT normalized (in range [0, 255]) while img is normalized 
    return yuv422, y, u, v, H, W

    

#only applicable to training datasets 
# read and STORE as YUV 422 f32 image  
def read_image_yuv422_f32(img):
    #results are a 3-channel image
    img = img * 255.0

    r, g, b = cv2.split(img)
    rows, cols = r.shape

    # Compute Y, U, V according to the formula described here:
    # https://developer.apple.com/documentation/accelerate/conversion/understanding_ypcbcr_image_formats
    # U applies Cb, and V applies Cr

    # Use BT.709 standard "full range" conversion formula
    y = 0.2126*r + 0.7152*g + 0.0722*b
    u = 0.5389*(b-y) + 128
    v = 0.6350*(r-y) + 128

    # Downsample u horizontally
    u = cv2.resize(u, (cols//2, rows), interpolation=cv2.INTER_LINEAR)

    # Downsample v horizontally
    v = cv2.resize(v, (cols//2, rows), interpolation=cv2.INTER_LINEAR)

    # Convert y to uint8 with rounding
    y = np.round(y).astype(np.uint8)

    # Convert u and v to uint8 with clipping and rounding: 
    u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
    v = np.round(np.clip(v, 0, 255)).astype(np.uint8)

    # Interleave u and v:
    uv = np.zeros_like(y)
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    # Merge y and uv channels
    yuv422 = cv2.merge((y, uv))
    #should have shape (800, 800, 2)
    
    img = yuv422.astype(np.float32) / 255.0
    #dividing by 255 here 

    # originally after the cv2.resize method for regular rgb images (i think for easier indexing when iterating thru dataset)
    
    #note that y, u, v are NOT normalized (in range [0, 255]) while img is normalized 
    return img, y, u, v, H, W


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