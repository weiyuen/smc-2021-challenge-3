import numpy as np

# color-class mapping for standardizing CARLA + Cityscape datasets according to ORNL SMC 2021 requirements

# (H*W*3) --> (H*W)
def color_to_class(image):  
    color_dict = {
        (70, 70, 70): 0,
        (190, 153, 153): 1,
        (153, 153, 153): 2,
        (244, 35, 232): 3,
        (107, 142, 35): 4,
        (102, 102, 156): 5,
        (128, 64, 128): 6,
        (157, 234, 50): 6,
        (250, 170, 30): 7,
        (220, 220, 0): 7,
        (220, 20, 60): 8,
        (255, 0, 0): 8,
        (0, 0, 142): 9,
        (0, 0, 70): 10,
        (0, 60, 100): 11,
        (0, 80, 100): 12,
        (0, 0, 230): 13,
        (119, 11, 32): 13
    }
    # get unique values and inverse matrix (used to rebuild original matrix)
    # reshape is used to get unique (r,g,b) pair values
    unique, inverse = np.unique(image.reshape(-1, image.shape[2]), axis=0, return_inverse=True)
    
    # key into dict for each unique (r,g,b) pair, then rebuild using inverse
    # 14 is default label
    mapped_image = np.array([color_dict.get(tuple(x.tolist()), 14) for x in unique])[inverse]
    
    # reshape into 2D image
    mapped_image = mapped_image.reshape(image.shape[:2])
    return mapped_image

# (H*W) --> (H*W*3)
def class_to_color(image):
    color_dict = {
        0: np.array([70, 70, 70], dtype=np.uint8),
        1: np.array([190, 153, 153], dtype=np.uint8),
        2: np.array([153, 153, 153], dtype=np.uint8),
        3: np.array([244, 35, 232], dtype=np.uint8),
        4: np.array([107, 142, 35], dtype=np.uint8),
        5: np.array([102, 102, 156], dtype=np.uint8),
        6: np.array([128, 64, 128], dtype=np.uint8),
        7: np.array([220, 220, 0], dtype=np.uint8),
        8: np.array([220, 20, 60], dtype=np.uint8),
        9: np.array([0, 0, 142], dtype=np.uint8),
        10: np.array([0, 0, 70], dtype=np.uint8),
        11: np.array([0, 60, 100], dtype=np.uint8),
        12: np.array([0, 80, 100], dtype=np.uint8),
        13: np.array([119, 11, 32], dtype=np.uint8),
        14: np.array([0, 0, 0], dtype=np.uint8)
    }
    
    image = np.expand_dims(image, axis=-1)
    # get unique values and inverse matrix (used to rebuild original matrix)
    unique, inverse = np.unique(image, return_inverse=True)
    
    # key into dict for each unique value, then rebuild using inverse
    mapped_image = np.array([color_dict.get(x) for x in unique])[inverse]
    
    # reshape into 2D image
    mapped_image = mapped_image.reshape((image.shape[0], image.shape[1], 3))
    return mapped_image