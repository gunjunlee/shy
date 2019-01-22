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
        col, row = sqrt, sqrt
    elif res <= -sqrt + math.sqrt(sqrt**2+sqrt):
        col, row = sqrt, sqrt+1
    else:
        col, row = sqrt+1, sqrt+1

    fig, axs = plt.subplots(nrows=row, ncols=col)
    axs = axs.reshape(col, row)

    for c in range(col):
        for r in range(row):
            if c*row + r > total - 1:
                break
            if c == 0 and r == 0:
                axs[0, 0].imshow(img)
                continue
            axs[c, r].imshow(args[c*row + r - 1])
    plt.show()


def check_integrity(fpath, md5=None):
    import hashlib

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
    if md5c != md5:
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
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True)))
        except:
            if url[:5] == 'https':
                url = url.replace('https:, http:')
                print('Failed download. Trying https -> http instead')
                print('Downloading '+url+' to '+fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True)))
