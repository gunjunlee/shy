import sys
import pdb
import matplotlib.pyplot as plt

def err_hook():
  """[summary]
  """

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
