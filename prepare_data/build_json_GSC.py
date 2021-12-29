from random import shuffle
import json
import os


def GSC_json(config):
        # build vocabulary
    if config.n_keyword == 35:
        vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                 'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
                 'backward', 'forward', 'follow', 'learn', 'visual']
    elif config.n_keyword == 12:
        vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                 'unknown', 'silence']
    else:
        raise ValueError('n_keyword must be 12 or 35')

    # read file path
    with open(os.path.join(config.data_dir, 'training_list.txt'), 'r') as training_file:
        training_set = [line.strip('\n') for line in training_file]
    
    with open(os.path.join(config.data_dir, 'validation_list.txt'), 'r') as validation_file:
        validation_set = [line.strip('\n') for line in validation_file]

    with open(os.path.join(config.data_dir, 'testing_list.txt'), 'r') as testing_file:
        testing_set = [line.strip('\n') for line in testing_file]

    with open(os.path.join(config.data_dir, 'background_list.txt'), 'r') as background_file:
        background_set = [line.strip('\n') for line in background_file]
    
    print(len(training_set), len(validation_set), len(testing_set), len(background_set))
    print(f'Building json file for {config.n_keyword} keywords ...')

    # create json file
    ftrain = open(os.path.join(config.data_dir, f'train_{config.n_keyword}.json'), 'w')
    fval = open(os.path.join(config.data_dir, f'validation_{config.n_keyword}.json'), 'w')
    ftest = open(os.path.join(config.data_dir, f'test_{config.n_keyword}.json'), 'w')

    # write file path to json
    for word in vocab:
        if word == 'unknown':
            unknown_word = ['backward', 'bed', 'bird', 'cat', 'dog',
                            'eight', 'five', 'follow', 'forward', 'four',
                            'happy', 'house', 'learn', 'marvin', 'nine',
                            'one', 'seven', 'sheila', 'six', 'three',
                            'tree', 'two', 'visual', 'wow', 'zero'] # 25 keywords
            for unk_word in unknown_word:
                count = 0
                for file in training_set:
                    if unk_word+'/' in file:
                        path =  os.path.join(os.path.join(config.data_dir, file))
                        line = {'file': path, 'text': word}
                        json.dump(line, ftrain)
                        ftrain.write('\n')
                        count += 1
                        if count == 120: # balanced data
                            break
            for unk_word in unknown_word:
                count = 0
                for file in validation_set:
                    if unk_word+'/' in file:
                        path =  os.path.join(os.path.join(config.data_dir, file))
                        line = {'file': path, 'text': word}
                        json.dump(line, fval)
                        fval.write('\n')
                        count += 1
                        if count == 15: # balanced data
                            break
            for unk_word in unknown_word:
                count = 0
                for file in testing_set:
                    if unk_word+'/' in file:
                        path =  os.path.join(os.path.join(config.data_dir, file))
                        line = {'file': path, 'text': word}
                        json.dump(line, ftest)
                        ftest.write('\n')
                        count += 1
                        if count == 15: # balanced data
                            break
        elif word == 'silence':
            shuffle(background_set)
            for file in background_set[:2800]:
                path =  os.path.join(os.path.join(config.data_dir, file))
                line = {'file': path, 'text': word}
                json.dump(line, ftrain)
                ftrain.write('\n')
            for file in background_set[2800:3200]:
                path =  os.path.join(os.path.join(config.data_dir, file))
                line = {'file': path, 'text': word}
                json.dump(line, fval)
                fval.write('\n')
            for file in background_set[3200:3600]:
                path =  os.path.join(os.path.join(config.data_dir, file))
                line = {'file': path, 'text': word}
                json.dump(line, ftest)
                ftest.write('\n')
        else:
            for file in training_set:
                if word+'/' in file:
                    path =  os.path.join(os.path.join(config.data_dir, file))
                    line = {'file': path, 'text': word}
                    json.dump(line, ftrain)
                    ftrain.write('\n')
            for file in validation_set:
                if word+'/' in file:
                    path =  os.path.join(os.path.join(config.data_dir, file))
                    line = {'file': path, 'text': word}
                    json.dump(line, fval)
                    fval.write('\n')
            for file in testing_set:
                if word+'/' in file:
                    path =  os.path.join(os.path.join(config.data_dir, file))
                    line = {'file': path, 'text': word}
                    json.dump(line, ftest)
                    ftest.write('\n')
    
    # close file
    ftrain.close()
    fval.close()
    ftest.close()

