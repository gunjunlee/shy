import sys
import pdb

def err_hook():
  sys_legacy_hook = sys.excepthook
  def new_hook(type, value, traceback):
    sys_legacy_hook(type, value, traceback)
    pdb.post_mortem(traceback)
  sys.excepthook = new_hook