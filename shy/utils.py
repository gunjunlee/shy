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