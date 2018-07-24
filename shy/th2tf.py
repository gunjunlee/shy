def th2tf(th_net, tf_net, th_pth=None, log=False):
    import torch
    import torch.nn as nn
    import tensorflow as tf
    if th_pth is not None:
        th_net.load_state_dict(torch.load(th_pth))
    
    m = {}
    for (k, v) in th_net.named_parameters():
        m[k] = v
    
    with tf.Session():
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            torch_name = v.name.replace('weights:0', 'weight') \
                               .replace('biases:0', 'bias') \
                               .replace('norm:0', 'norm.weight') \
                               .replace('/', '.')
            if log: print(v.name, v.get_shape(), torch_name, m[torch_name].size())

            if len(v.get_shape()) == 4:
                v.load(m[torch_name].permute(2, 3, 1, 0).data.numpy())
            elif len(v.get_shape()) == 1:
                v.load(m[torch_name].data.numpy())
            else:
                raise Exception("unknown shape")
        if log: print('transfer done')
