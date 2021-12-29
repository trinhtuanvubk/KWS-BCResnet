from torch.utils.data import Dataset
import torch
import json
import torchaudio


class GSC_Dataset(Dataset):

    def __init__(self, config, stage='train'):
        self.config = config
        self.stage = stage
        self.data, self.vocab = self.load_json()

    def load_json(self):
        data = []
        with open(f'{self.config.data_dir}/{self.stage}_{self.config.n_keyword}.json') as file:
            for line in file:
                data.append(json.loads(line))

        if self.config.n_keyword == 35:
            vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                     'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                     'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
                     'backward', 'forward', 'follow', 'learn', 'visual']
        elif self.config.n_keyword == 12:
            vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                     'unknown', 'silence']
        else:
            raise ValueError('n_keyword must be 12 or 35')

        return data, vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # raw audio and label respectively
        audio, _ = torchaudio.load(self.data[idx]['file'])
        label = self.vocab.index(self.data[idx]['text'])
        pad_audio = padding(audio)
        length = torch.tensor(audio.shape[1] / pad_audio.shape[1])
        return pad_audio.squeeze(0), length, label



def padding(x):
    if x.shape[1] >= 16000:
        x = x[:,:16000]
    else:
        pad = torch.zeros(1,16000)
        pad[:,:x.shape[1]] = x
        x = pad
    return x