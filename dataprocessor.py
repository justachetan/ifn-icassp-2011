import os
import random
import pickle
import argparse

import parselmouth
import numpy as np


def get_ref_data_audio(speaker_dirs):
    neu_ref_audio_files = list()
    for speaker_dir in speaker_dirs:
        files = [speaker_dir + "/" + j for j in os.listdir(speaker_dir)]
        neu_files = [i for i in files if i.split("-")[3] == "01"]
        neu_file = random.choice(neu_files)
        neu_ref_audio_files.append(neu_file)

    return neu_ref_audio_files


def get_pitch_cntr(sound_file_path):
    sound = parselmouth.Sound(sound_file_path)
    pitch = sound.to_pitch_ac()
    f0 = pitch.selected_array['frequency']
    return f0


def preprocess_ref(ref_files, save=None):
    ref_corpus = list()
    for file in ref_files:
        f0_cntr = get_pitch_cntr(file)
        ref_corpus.append((file, f0_cntr))

    if save is not None:
        with open(save, "wb") as f:
            pickle.dump(ref_corpus, f)

    return ref_corpus


def preprocess_emo(emo_files, save=None):
    emo_corpus = list()
    for file in emo_files:
        f0_cntr = get_pitch_cntr(file)
        speaker = file.split("/")[-2]
        emo_label = file.split("-")[3]
        if emo_label != "01":
            emo_label = 0  # emotional
        else:
            emo_label = 1  # neutral
        emo_corpus.append((file, f0_cntr, speaker, emo_label))

    if save is not None:
        with open(save, "wb") as f:
            pickle.dump(emo_corpus, f)

    return emo_corpus

parser = argparse.ArgumentParser(
    description='Preprocessing script for the RAVDESS dataset for IFN input')

parser.add_argument('--data_dir', type=str,
                    help='Main directory of the RAVDESS dataset')
parser.add_argument('--out_emo', type=str,
                    help='Path where emotional corpus will be saved', default="./RAVDESS_emo.pkl")
parser.add_argument('--out_ref', type=str,
                    help='Path where neutral reference corpus will be saved', default="./RAVDESS_ref.pkl")

args = parser.parse_args()

DATA_DIR = args.data_dir

speaker_dirs = [DATA_DIR + i for i in os.listdir(DATA_DIR)]

speaker_dirs = sorted(speaker_dirs, key=lambda x: x.split("_")[1])

ref_audio_files = get_ref_data_audio(speaker_dirs)

aud_data_files = [
    i + "/" + j for i in speaker_dirs for j in os.listdir(i) if i + "/" + j not in ref_audio_files]


preprocess_ref(ref_audio_files, args.out_ref)
preprocess_emo(aud_data_files, args.out_emo)
