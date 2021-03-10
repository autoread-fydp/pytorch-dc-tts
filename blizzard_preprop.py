"""credit: https://github.com/keithito/tacotron/blob/master/datasets/blizzard.py"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from hparams import HParams as hp
import audio

from tqdm import tqdm
from multiprocessing import cpu_count

_max_out_length = 700
_end_buffer = 0.05
_min_confidence = 90

# Note: "A Tramp Abroad" & "The Man That Corrupted Hadleyburg" are higher quality than the others.
books = [
    'ATrampAbroad',
    'TheManThatCorruptedHadleyburg',
    # 'LifeOnTheMississippi',
    # 'TheAdventuresOfTomSawyer',
]


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    for book in books:
        with open(os.path.join(in_dir, book, 'sentence_index.txt')) as f:
            for line in f:
                parts = line.strip().split('\t')
                if line[0] is not '#' and len(parts) == 8 and float(parts[3]) > _min_confidence:
                    wav_path = os.path.join(in_dir, book, 'wav', '%s.wav' % parts[0])
                    labels_path = os.path.join(in_dir, book, 'lab', '%s.lab' % parts[0])
                    text = parts[5]
                    task = partial(_process_utterance, out_dir, index, wav_path, labels_path, text)
                    futures.append(executor.submit(task))
                    index += 1
    results = [future.result() for future in tqdm(futures)]
    return [r for r in results if r is not None]


def _process_utterance(out_dir, index, wav_path, labels_path, text):
    # Load the wav file and trim silence from the ends:
    wav = audio.load_wav(wav_path)
    start_offset, end_offset = _parse_labels(labels_path)
    start = int(start_offset * hp.sr)
    end = int(end_offset * hp.sr) if end_offset is not None else -1
    wav = wav[start:end]
    max_samples = _max_out_length * hp.frame_shift * hp.sr
    if len(wav) > max_samples:
        # print(wav_path + ": wav too long")
        return None
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    spectrogram_filename = 'blizzard-spec-%05d.npy' % index
    mel_filename = 'blizzard-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    return (spectrogram_filename, mel_filename, n_frames, text)


def _parse_labels(path):
    labels = []
    with open(os.path.join(path)) as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 3:
                labels.append((float(parts[0]), ' '.join(parts[2:])))
    start = 0
    end = None
    if labels[0][1] == 'sil':
        start = labels[0][0]
    if labels[-1][1] == 'sil':
        end = labels[-2][0] + _end_buffer
    return (start, end)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    hours = frames * hp.frame_shift / 3600
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    in_dir = "datasets/blizzard-taco-preprop"
    out_dir = "datasets/blizzard-taco-processed"
    num_workers = cpu_count()
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)
