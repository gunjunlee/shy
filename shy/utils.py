try:
    from PIL import Image
    if "post" not in Image.PILLOW_VERSION:
        print("[shy warning] You have /pillow/ instead of /pillow-simd/.")
except:
    pass

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def err_hook():
    import sys
    import bdb
    import pdb
    import backtrace

    backtrace.hook(align=True)

    if run_from_ipython():
        import IPython
        old_showtraceback = IPython.core.interactiveshell.InteractiveShell.showtraceback

        def new_showtraceback(exc_tuple=None, filename=None, tb_offset=None, exception_only=False, running_compiled_code=False):
            old_showtraceback(exc_tuple, filename, tb_offset, exception_only, running_compiled_code)
            try:
                import ipdb
            except:
                print('[shy warning]: You do not have /ipdb/. Please install /ipdb/')
                return
            ipdb.post_mortem(tb_offset)

        IPython.core.interactiveshell.InteractiveShell.showtraceback = new_showtraceback
    else:
        old_hook = sys.excepthook

        def new_hook(type_, value, traceback):
            old_hook(type_, value, traceback)
            if type_ != KeyboardInterrupt and type_ != bdb.BdbQuit:
                pdb.post_mortem(traceback)
        sys.excepthook = new_hook


def show_img(img, show=True):
    import matplotlib.pyplot as plt

    plt.imshow(img)
    if show:
        plt.show()


def show_imgs(img, *args, nrows=None, ncols=None, show=True):
    if len(args) == 0:
        return show_img(img)

    import matplotlib.pyplot as plt
    import math

    total = 1 + len(args)
    if nrows is not None and ncols is not None:
        assert nrows * ncols >= total, f'nrows: {nrows}, ncols: {ncols}, # of images: {total}. nrows * ncols (={ncols*nrows}) is lower than number of images!'
    elif nrows is not None:
        ncols = int(math.ceil(total / nrows))
    elif ncols is not None:
        nrows = int(math.ceil(total / ncols))
    else:
        sqrt = int(math.sqrt(total))
        res = math.sqrt(total) - sqrt

        if res == 0:
            nrows, ncols = sqrt, sqrt
        elif res <= -sqrt + math.sqrt(sqrt**2 + sqrt):
            nrows, ncols = sqrt, sqrt + 1
        else:
            nrows, ncols = sqrt + 1, sqrt + 1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    axs = axs.reshape(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            if r * ncols + c > total - 1:
                axs[r, c].axis('off')
                continue
            if r == 0 and c == 0:
                axs[0, 0].imshow(img)
                continue
            axs[r, c].imshow(args[r * ncols + c - 1])

    if show:
        plt.show()


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
    import urllib.request
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
        print('File already exists: ' + fpath)
    elif os.path.isfile(fpath):
        print('File already exists... but not match md5: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            with tqdm(total=None) as pbar:
                urllib.request.urlretrieve(url, fpath,
                                           reporthook=gen_bar_updater(pbar))
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead')
                print('Downloading ' + url + ' to ' + fpath)
            with tqdm(total=None) as pbar:
                urllib.request.urlretrieve(url, fpath,
                                           reporthook=gen_bar_updater(pbar))
        print('Download completed: ' + url + ' to ' + fpath)
        if not check_integrity(fpath, md5):
            print('not match md5: ' + fpath)


def _animation(evt):
    import time
    import sys
    frame = ['.', '..', '...', '....']
    idx = 0
    while not evt.is_set():
        sys.stdout.write(frame[idx])
        sys.stdout.flush()
        time.sleep(0.5)
        sys.stdout.write('\b' * len(frame[idx]))
        sys.stdout.write(' ' * len(frame[idx]))
        sys.stdout.write('\b' * len(frame[idx]))
        idx += 1
        if idx == len(frame):
            idx = 0

def loading(func, args=(), kwargs={}, verbose=True, desc=None):
    import threading

    evt = threading.Event()
    loading_ani_p = threading.Thread(target=_animation, args=(evt,))

    if desc is not None:
        pre_desc, post_desc = desc.split('{bar}')

    if verbose:
        if desc is not None:
            print(pre_desc, end='')
        else:
            print('loading', end='')
    loading_ani_p.start()
    result = func(*args, **kwargs)
    evt.set()
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
