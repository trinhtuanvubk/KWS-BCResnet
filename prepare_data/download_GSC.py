import numpy as np
import soundfile
import os
from config.config import Config


def download_GSC(config):
    
    # download data
    if not os.path.exists(os.path.join(config.data_dir, 'speech_commands_v0.02.tar.gz')):
        print('Downloading GSC dataset ...')
        os.system(f'wget -P {config.data_dir} http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')
    else:
        print('Skipping download GSC dataset ...')
    # extract data
    if config.extract:
        print('Extracing GSC dataset ...')
        os.system(f'tar -xf {os.path.join(config.data_dir, "speech_commands_v0.02.tar.gz")} -C {config.data_dir}')
        print('Extracting RIR and Noise dataset ...')
        os.system(f'unzip -n -q {os.path.join(config.data_dir, "rirs_noises.zip")} -d {config.data_dir}')
    else:
        print('Skipping extract dataset ...')

    print('Preparing dataset ...')
    # get validation list
    with open(os.path.join(config.data_dir, 'validation_list.txt'), 'r') as validation_file:
        validation_set = [line.strip('\n') for line in validation_file]
    # get test list
    with open(os.path.join(config.data_dir, 'testing_list.txt'), 'r') as testing_file:
        testing_set = [line.strip('\n') for line in testing_file]

    # get data list
    keyword = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
               'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
               'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
               'backward', 'forward', 'follow', 'learn', 'visual']
    
    data_set = []
    for word in keyword:
        word_file = [os.path.join(word, file) for file in os.listdir(os.path.join(config.data_dir, word))]
        data_set += word_file
    
    # training list = data list - validation list - test list
    training_set = set(data_set).symmetric_difference(set(validation_set)).symmetric_difference(set(testing_set))
    training_set = sorted(list(training_set))

    # write training list to file
    open(os.path.join(config.data_dir, 'training_list.txt'), 'w')
    with open(os.path.join(config.data_dir, 'training_list.txt'), 'a') as fin:
        [fin.write(path + '\n') for path in training_set]
    
    # processing background noise
    background_list = os.listdir(os.path.join(config.data_dir, '_background_noise_'))
    background_set = [file for file in background_list if file.endswith('.wav')] # 6 files
    for file in background_set:
        folder_path = os.path.join(config.data_dir, '_background_noise_')
        file_path = os.path.join(folder_path, file)
        os.makedirs(file_path.replace('.wav',''), exist_ok=True)
        signal, rate = soundfile.read(file_path)
        for i in range(600): # 6*600=3600 to balancing data
            offset = np.random.randint(0, len(signal)-rate-1)
            sub_signal = signal[offset:offset+rate]
            soundfile.write(os.path.join(folder_path, file.replace('.wav',''), file.replace('.wav','')+'_'+str(i)+'.wav'), sub_signal, rate)
    
    # write background list to file
    open(os.path.join(config.data_dir, 'background_list.txt'), 'w')
    for root, dirs, _ in os.walk(os.path.join(config.data_dir, '_background_noise_')):
        for folder in dirs:
            files = os.listdir(os.path.join(root, folder))
            with open(os.path.join(config.data_dir, 'background_list.txt'), 'a') as fin:
                [fin.write(os.path.join('_background_noise_', folder, file) + '\n') for file in files]
        break

