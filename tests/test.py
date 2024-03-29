from PIL import Image
import matplotlib.pyplot as plt

import os
import os.path as osp

CUR_DIR = osp.dirname(__file__)

import imp

imp.load_source('shy', osp.join(CUR_DIR, '../shy/__init__.py'))
import shy

cache_path = osp.join(CUR_DIR, 'izone_chaewon.jpg')
pkl_path = osp.join(CUR_DIR, 'izone_chaewon.pkl')

# download % show_img(s)
try:
    # md5 match
    shy.download_url('https://www.dropbox.com/s/jcfyl8zsrlekgh6/izone_chaewon.jpg?dl=1',
                     cache_path,
                     '9abd027e11a62b3c7050c83c669e72b0')
    # md5 not match
    shy.download_url('https://www.dropbox.com/s/jcfyl8zsrlekgh6/izone_chaewon.jpg?dl=1',
                     cache_path,
                     '124')

    # check integrity
    assert shy.check_integrity(cache_path, '9abd027e11a62b3c7050c83c669e72b0'), 'check_integrity failed'
    assert shy.check_integrity(cache_path, '9abd027e11a6'), 'check_integrity failed'
    assert not shy.check_integrity(cache_path, '124'), 'check_integrity failed'

    img = Image.open(cache_path)

    # show_img
    shy.show_img(img)

    # show_imgs
    for i in range(1, 4):
        ar = [img for _ in range(1, i)]
        shy.show_imgs(img, *ar)

    # draw bbox
    img = img.resize((100, 100))
    shy.show_image(img)

    shy.save_pkl(img, pkl_path)
    img = shy.load_pkl(pkl_path)
    shy.show_image(img)

finally:
    if osp.isfile(cache_path):
        os.remove(cache_path)
    if osp.isfile(pkl_path):
        os.remove(pkl_path)


def poo(num):
    for i in range(num):
        pass
    return True

assert shy.loading(poo, args=(int(2e+8),)), 'loading return value error'


def poo(num):
    for i in range(num):
        pass
    return

shy.loading(poo, args=(int(2e+8),), desc='calc{bar} is done!')

shy.err_hook()
a = 1/0
