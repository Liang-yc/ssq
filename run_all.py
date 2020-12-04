import tensorflow as tf
tf.compat.v1.disable_eager_execution()
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)


if __name__ == '__main__':

    from ssq4all_red import run_training as a,FLAGS
    a()
    del_all_flags(FLAGS)
    from ssq4all import run_training as b,FLAGS
    b()
    del_all_flags(FLAGS)
    from ssq4all_v2 import run_training as c,FLAGS
    c()
    del_all_flags(FLAGS)
    from ssq4all_v4 import run_training as d,FLAGS
    d()
    del_all_flags(FLAGS)