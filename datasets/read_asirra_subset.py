import os
import numpy as np
from imageio import imread, imsave
from skimage.transform import resize

def read_asirra_subset(subset_dir, one_hot=True, sample_size=None):
    """
    디스크로부터 Asirra Dogs vs. Cats 데이터셋을 로드하고,
    AlexNet을 학습하는 데 사용하기 위한 형태로 전처리를 수행함.
    :param subset_dir: str, 원본 데이터셋이 저장된 디렉터리 경로.
    :param one_hot: bool, one-hot 인코딩 형태의 레이블을 반환할 것인지 여부.
    :param sample_size: int, 전체 데이터셋을 모두 사용하지 않는 경우, 사용하고자 하는 샘플 이미지 개수.
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, num_channels) or (N,).
    """
    # 학습+검증 데이터셋을 읽어들임
    filename_list = os.listdir(subset_dir)
    set_size = len(filename_list)

    if sample_size is not None and sample_size < set_size:
        # sample_size가 명시된 경우, 원본 중 일부를 랜덤하게 샘플링함
        filename_list = np.random.choice(filename_list, size=sample_size, replace=False)
        set_size = sample_size
    else:
        # 단순히 filename list의 순서를 랜덤하게 섞음
        np.random.shuffle(filename_list)

    # 데이터 array들을 메모리 공간에 미리 할당함
    X_set = np.empty((set_size, 256, 256, 3), dtype=np.float32)    # (N, H, W, 3)
    y_set = np.empty((set_size), dtype=np.uint8)                   # (N,)
    for i, filename in enumerate(filename_list):
        if i % 1000 == 0:
            print('Reading subset data: {}/{}...'.format(i, set_size), end='\r')
        label = filename.split('.')[0]
        if label == 'cat':
            y = 0
        else:  # label == 'dog'
            y = 1
        file_path = os.path.join(subset_dir, filename)
        img = imread(file_path)    # shape: (H, W, 3), range: [0, 255]
        img = resize(img, (256, 256), mode='constant').astype(np.float32)    # (256, 256, 3), [0.0, 1.0]
        X_set[i] = img
        y_set[i] = y

    if one_hot:
        # 모든 레이블들을 one-hot 인코딩 벡터들로 변환함, shape: (N, num_classes)
        y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_set = y_set_oh
    print('\nDone')

    return X_set, y_set

def random_crop_reflect(images, crop_l):
    """
    Perform random cropping and reflection from images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        # Randomly crop patch
        y = np.random.randint(H-crop_l)
        x = np.random.randint(W-crop_l)
        image = image[y:y+crop_l, x:x+crop_l]    # (h, w, C)

        # Randomly reflect patch horizontally
        reflect = bool(np.random.randint(2))
        if reflect:
            image = image[:, ::-1]

        augmented_images.append(image)
    return np.stack(augmented_images)    # shape: (N, h, w, C)

def corner_center_crop_reflect(images, crop_l):
    """
    Perform 4 corners and center cropping and reflection from images,
    resulting in 10x augmented patches.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, 10, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        aug_image_orig = []
        # Crop image in 4 corners
        aug_image_orig.append(image[:crop_l, :crop_l])
        aug_image_orig.append(image[:crop_l, -crop_l:])
        aug_image_orig.append(image[-crop_l:, :crop_l])
        aug_image_orig.append(image[-crop_l:, -crop_l:])
        # Crop image in the center
        aug_image_orig.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                                    W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
        aug_image_orig = np.stack(aug_image_orig)    # (5, h, w, C)

        # Flip augmented images and add it
        aug_image_flipped = aug_image_orig[:, :, ::-1]    # (5, h, w, C)
        aug_image = np.concatenate((aug_image_orig, aug_image_flipped), axis=0)    # (10, h, w, C)
        augmented_images.append(aug_image)
    return np.stack(augmented_images)    # shape: (N, 10, h, w, C)


def center_crop(images, crop_l):
    """
    Perform center cropping of images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    cropped_images = []
    for image in images:    # image.shape: (H, W, C)
        # Crop image in the center
        cropped_images.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                              W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
    return np.stack(cropped_images)
