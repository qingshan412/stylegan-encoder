import os
import sys
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def align_file(src, tgt):
    for img_name in os.listdir(src):
        print('processing', img_name)
        raw_img_path = os.path.join(src, img_name)
        face_landmarks = landmarks_detector.get_landmarks(raw_img_path)[0]
        aligned_face_path = os.path.join(tgt, img_name)
        image_align(raw_img_path, aligned_face_path, face_landmarks)

if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    # landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
    #                                            LANDMARKS_MODEL_URL, cache_subdir='temp'))
    landmarks_model_path = 'cache/shape_predictor_68_face_landmarks.dat'

    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]

    os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)

    landmarks_detector = LandmarksDetector(landmarks_model_path)

    if 'divided' in RAW_IMAGES_DIR:
        for folder_name in os.listdir(RAW_IMAGES_DIR):
            os.makedirs(ALIGNED_IMAGES_DIR + os.sep + folder_name, exist_ok=True)
            print('processing', folder_name)
                for img_name in os.listdir(RAW_IMAGES_DIR + os.sep + folder_name):
                    align_file(RAW_IMAGES_DIR + os.sep + folder_name, ALIGNED_IMAGES_DIR + os.sep + folder_name)
    else:
        align_file(RAW_IMAGES_DIR, ALIGNED_IMAGES_DIR)

        # for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
        #     face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
        #     aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)

        #     image_align(raw_img_path, aligned_face_path, face_landmarks)
