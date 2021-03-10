"""Data loader for the LJSpeech dataset. See: https://keithito.com/LJ-Speech-Dataset/"""
import os
import re
import codecs
import unicodedata
import numpy as np

from torch.utils.data import Dataset

from .cleaners import english_cleaners
from .autoread import vocab, char2idx, idx2char, text_normalize, get_test_data


def read_metadata(metadata_file):
    mag_fnames, mel_fnames, text_lengths, texts = [], [], [], []
    transcript = os.path.join(metadata_file)
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for line in lines:
        try:
            mag_fname, mel_fname, n_frames, text = line.strip().split("|")
        except:
            print(line)
            raise ValueError

        mag_fnames.append(mag_fname)
        mel_fnames.append(mel_fname)

        text = text_normalize(text) + "E"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.long))

    return mag_fnames, mel_fnames, text_lengths, texts


class Blizzard(Dataset):
    def __init__(self, keys, dir_name='blizzard-taco-processed'):
        self.keys = keys
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        self.mag_fnames, self.mel_fnames, self.text_lengths, self.texts = read_metadata(os.path.join(self.path, 'train.txt'))

    def slice(self, start, end):
        self.mag_fnames = self.mag_fnames[start:end]
        self.mel_fnames = self.mel_fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]

    def __len__(self):
        return len(self.mel_fnames)

    def __getitem__(self, index):
        data = {}
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'mels' in self.keys:
            # (39, 80)
            data['mels'] = np.load(os.path.join(self.path, self.mel_fnames[index]))
        if 'mags' in self.keys:
            # (39, 80)
            data['mags'] = np.load(os.path.join(self.path, self.mag_fnames[index]))
        if 'mel_gates' in self.keys:
            data['mel_gates'] = np.ones(data['mels'].shape[0], dtype=np.int)  # TODO: because pre processing!
        if 'mag_gates' in self.keys:
            data['mag_gates'] = np.ones(data['mags'].shape[0], dtype=np.int)  # TODO: because pre processing!
        return data
