from PIL import Image
import matplotlib.pyplot as plt

import os
import os.path as osp

import shy

CUR_DIR = osp.dirname(__file__)

cache_path = osp.join(CUR_DIR, 'izone_chaewon.jpg')

# download
## md5 match
shy.download_url('https://www.dropbox.com/s/jcfyl8zsrlekgh6/izone_chaewon.jpg?dl=1',
                 cache_path,
                 '9abd027e11a62b3c7050c83c669e72b0')
## md5 not match
shy.download_url('https://www.dropbox.com/s/jcfyl8zsrlekgh6/izone_chaewon.jpg?dl=1',
                 cache_path,
                 '124')


# check integrity
assert shy.check_integrity(cache_path, '9abd027e11a62b3c7050c83c669e72b0'), 'check_integrity failed'
assert not shy.check_integrity(cache_path, '124'), 'check_integrity failed'


img = Image.open(cache_path)

# show_img
shy.show_img(img)

# show_imgs
for i in range(1, 4):
    ar = [img for _ in range(1, i)]
    shy.show_imgs(img, *ar)

# shy.err_hook()
# a = 1/0

os.remove(cache_path)
