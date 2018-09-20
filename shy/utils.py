import matplotlib.pyplot as plt

def err_hook():
  """[summary]
  """
  import sys
  import pdb
  sys_legacy_hook = sys.excepthook
  def new_hook(type, value, traceback):
    sys_legacy_hook(type, value, traceback)
    pdb.post_mortem(traceback)
  sys.excepthook = new_hook

def show_img(img):
  """[summary]
  
  Args:
    img ([type]): [description]
  """

  plt.imshow(img)
  plt.show()

def mv_prob(from_folder, to_folder, prob):
  """move files from 'from_folder' to 'to_folder' with probability 'prob'
  
  Args:
    from_folder (str): from folder
    to_folder (str): to folder
  """

  import random
  import os
  import shutil
  for path, _, files in os.walk(from_folder):
    for file_ in files:
      if random.uniform(0, 1) < prob:
        from_path = os.path.join(path, file_)
        to_path = from_path.replace(from_folder, to_folder, 1)
        dir, _ = os.path.split(to_path)
        if not os.path.exists(dir):
          os.makedirs(dir)
        shutil.move(from_path, to_path)

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

def download_url(url, root, filename, md5):
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

  root = os.path.expanduser(root)
  fpath = os.path.join(root, filename)

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