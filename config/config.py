class Config(object):
    data_dir = './data'
    extract = True
   
    sr = 16000
    win_length = 25
    hop_length = 10
    filter_shape = 'triangular'
    n_mels = 40
    n_fft = 400

    n_embed = 512
    n_keyword = 12 

    m = 0.5 
    s = 64
 
    batch_size = 32
    shuffle = True
    pin_memory = True
    num_epoch = 2
    log_iter = 10 
    num_worker = 1
    clip_grag_norm = 0.5
    limit_train_batch = -1
    limit_val_batch = -1

    top_k = 5
    plot = True

