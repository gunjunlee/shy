def err_hook():
    import sys
    import pdb
    import backtrace

    backtrace.hook(align=True)

    old_hook = sys.excepthook

    def new_hook(type_, value, traceback):
        old_hook(type_, value, traceback)
        if type_ != KeyboardInterrupt:
            pdb.post_mortem(traceback)
    sys.excepthook = new_hook


def show_img(img):
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()


def show_imgs(img, *args):
    if len(args) == 0:
        return show_img(img)

    import matplotlib.pyplot as plt
    import math

    total = 1 + len(args)
    sqrt = int(math.sqrt(total))
    res = math.sqrt(total) - sqrt

    if res == 0:
        row, col = sqrt, sqrt
    elif res <= -sqrt + math.sqrt(sqrt**2+sqrt):
        row, col = sqrt, sqrt+1
    else:
        row, col = sqrt+1, sqrt+1

    fig, axs = plt.subplots(nrows=row, ncols=col)
    axs = axs.reshape(row, col)

    for r in range(row):
        for c in range(col):
            if r*col + c > total - 1:
                axs[r, c].axis('off')
                continue
            if r == 0 and c == 0:
                axs[0, 0].imshow(img)
                continue
            axs[r, c].imshow(args[r*col + c - 1])
    plt.show()


def draw_bbox(img, bbox, **kwarg):
    '''draw bbox on copied img

    Parameters
    ----------
    img : PIL.Image
        image which you want to draw box on
    bbox : iterable
        [x_min, y_min, x_max, y_max]
        normalized coordinate. [0, 1]

    Returns
    -------
    c_img: PIL.Image
        drawn copied image
    '''

    from PIL import ImageDraw

    img = img.copy()
    w, h = img.size

    draw = ImageDraw.Draw(img)
    draw.rectangle([bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h], **kwarg)
    del draw

    return img


def draw_bboxes(img, bboxes, **kwarg):
    '''draw bboxes on copied img

    Parameters
    ----------
    img : PIL.Image
        image which you want to draw box on
    bboxes : iterable
        [[x_min, y_min, x_max, y_max], ...]
        normalized coordinate. [0, 1]

    Returns
    -------
    c_img: PIL.Image
        drawn copied image
    '''

    from PIL import ImageDraw

    img = img.copy()
    w, h = img.size

    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle([bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h], **kwarg)
    del draw

    return img


def check_integrity(fpath, md5=None):
    import hashlib
    import os
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False

    md5o = hashlib.md5()

    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c[:len(md5)] != md5:
        return False
    return True


def download_url(url, path, md5=None):
    import urllib
    import os
    import errno
    from tqdm import tqdm

    def gen_bar_updater(pbar):
        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

    fpath = os.path.expanduser(path)
    root = os.path.dirname(fpath)

    if root != '':
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('File already exists: '+fpath)
    elif os.path.isfile(fpath):
        print('File already exists... but not match md5: '+fpath)
    else:
        try:
            print('Downloading '+url+' to '+fpath)
            with tqdm(total=None) as pbar:
                urllib.request.urlretrieve(url, fpath,
                                           reporthook=gen_bar_updater(pbar))
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead')
                print('Downloading '+url+' to '+fpath)
            with tqdm(total=None) as pbar:
                urllib.request.urlretrieve(url, fpath,
                                           reporthook=gen_bar_updater(pbar))
        print('Download completed: '+url+' to '+fpath)
        if not check_integrity(fpath, md5):
            print('not match md5: '+fpath)


def loading(func, args=(), kwargs={}, verbose=True, desc=None):
    import time
    import sys
    import multiprocessing as mp

    def animation(chk):
        frame = ['.', '..', '...', '....']
        idx = 0
        while True:
            sys.stdout.write(frame[idx])
            sys.stdout.flush()
            time.sleep(0.5)
            sys.stdout.write('\b'*len(frame[idx]))
            sys.stdout.write(' '*len(frame[idx]))
            sys.stdout.write('\b'*len(frame[idx]))
            idx += 1
            if idx == len(frame):
                idx = 0
            if chk.value == 0:
                break

    chk = mp.Value('i', 1)
    loading_ani_p = mp.Process(target=animation, args=(chk,))

    if desc is not None:
        pre_desc, post_desc = desc.split('{bar}')

    if verbose:
        if desc is not None:
            print(pre_desc, end='')
        else:
            print('loading', end='')
    loading_ani_p.start()
    result = func(*args, **kwargs)
    chk.value = 0
    loading_ani_p.join()
    if verbose:
        if desc is not None:
            print(post_desc)
        else:
            print(' completed!')
    return result

def save_pkl(obj, path):
    import pickle
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    import pickle

    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res