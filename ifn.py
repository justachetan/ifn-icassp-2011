import os
import sys
import random
import pickle
import argparse

import parselmouth
import numpy as np

from sklearn import mixture, discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import features


class IFN(object):

    def __init__(self, ref_corpus, enable_neutral, feat, max_iter=10,
                 ss_iter=400, neutral_label=1, emo_label=0, analysis=True,
                 analysis_test_size=0.3, norm_scheme="ifn", neu_threshold=0.7,
                 spkr_file_threshold=0.2, switch_threshold=0.05, log_dir="./test/",
                 ref_gmms=None, ldc=None):

        self.enable_neutral = enable_neutral
        self.ref_corpus = ref_corpus  # should be a list of ref f0 contours
        self.ref_feat = None
        self.features_ = [features.feature_map[i] for i in feat]
        self.neutral_label = neutral_label
        self.emo_label = emo_label

        self.X_train_raw = None
        self.y_train_raw = None

        # This flag will trigger treatment of training corpus as both
        # training and validation. Fit method will calculate stats of
        # validation set after every ss_iter and save the best model.
        # The results of every ss_iter will also be saved.
        self.analysis = analysis

        # Logs to store all necessary results
        self.log_dir = log_dir

        self.analysis_test_size = analysis_test_size

        self.ref_gmms = list()
        if ref_gmms is not None:
            self.ref_gmms = ref_gmms

        if self.enable_neutral:

            def fit_ref(self):

                X_ref = features.feat_ext(samples, self.features_)

                self.ref_feat = X_ref.copy()

                y_ref = np.array([self.neutral_label] * len(samples))

                for i in range(len(self.features_)):
                    gmm = GaussianMixture(n_components=2)
                    gmm.fit(X_ref[:, i].reshape(X_ref.shape[0], 1), y_ref)

                    self.ref_gmms.append(gmm)

        self.ss_iter = ss_iter
        self.max_iter = max_iter
        self.balance = False

        self.norm_constants = dict()
        self.norm_scheme = norm_scheme
        self.neu_threshold = neu_threshold

        self.spkr_file_threshold = spkr_file_threshold
        self.switch_threshold = switch_threshold

        self.best_ldc = ldc
        self.ldc = ldc

    def fit(self, X_train_raw, y_train_raw, spkrs, save=None):

        # X_train_raw should be a list of pitch contours
        # y_train_raw should be np array of labels
        # spkrs should be an np array of speakers
        self.X_train_raw = X_train_raw
        self.y_train_raw = y_train_raw
        self.spkrs = spkrs

        num_neu_samples = len(y_train_raw[y_train_raw == self.neutral_label])
        num_emo_samples = len(y_train_raw[y_train_raw != self.neutral_label])

        diff = np.abs(num_emo_samples - num_neu_samples) / num_neu_samples

        if diff > 0.1:
            self.balance = True
        else:
            self.ss_iter = 1

        for ss in range(self.ss_iter):

            X_train_raw_ss = self.X_train_raw
            y_train_raw_ss = self.y_train_raw
            spkrs_ss = self.spkrs

            if self.balance:

                # get indices of emotional samples in training set
                emo_idx = [i for i in range(len(y_train_raw_ss)) if y_train_raw_ss[
                    i] != self.neutral_label]

                # get indices of neutral samples in training set
                neu_idx = [i for i in range(len(y_train_raw_ss)) if y_train_raw_ss[
                    i] == self.neutral_label]

                neutral_samples_per_speaker = dict()

                for i in neu_idx:
                    if spkrs_ss[i] not in neutral_samples_per_speaker:
                        neutral_samples_per_speaker[spkrs_ss[i]] = 0

                    neutral_samples_per_speaker[spkrs_ss[i]] += 1

                # print(emo_idx, neu_idx)

                # sample population of emotional samples equal in size to neutral samples for each speaker
                # emo_idx_sample = sum([np.random.choice([emo_idx[j] for i in np.unique(spkrs_ss) \
                #                                    for j in emo_idx if spkrs_ss[j] == i], \
                # neutral_samples_per_speaker[i]).tolist()], [])

                emo_idx_sample = list()

                for i in np.unique(spkrs_ss):

                    tmp = list()

                    for j in emo_idx:
                        # print(spkrs_ss[j], i)
                        if spkrs_ss[j] == i:

                            tmp.append(j)

                    # print(tmp)
                    emo_idx_sample.extend(np.random.choice(
                        tmp, neutral_samples_per_speaker[i]).tolist())

                # make undersampled dataset
                X_train_raw_ss = [X_train_raw_ss[j]
                                  for j in emo_idx_sample + neu_idx]
                y_train_raw_ss = np.array([self.emo_label] * len(emo_idx_sample)
                                          + [self.neutral_label] * len(neu_idx))
                spkrs_ss = np.array([spkrs_ss[i]
                                     for i in emo_idx_sample + neu_idx])

            # print(len(X_train_raw_ss), np.unique(
            #     np.vstack((y_train_raw_ss, spkrs_ss)).T, axis=0))

            if self.analysis:
                X_train_raw, X_test_raw, y_train_raw, y_test_raw, spkrs_train, spkrs_test = train_test_split(X_train_raw_ss,
                                                                                                             y_train_raw_ss,
                                                                                                             spkrs_ss,
                                                                                                             test_size=0.35,
                                                                                                             stratify=np.vstack((y_train_raw_ss, spkrs_ss)).T)

            self.norm_constants = {i: 1 for i in np.unique(spkrs_ss)}

            X_train_raw_cur = X_train_raw.copy()
            X_test_raw_cur = X_test_raw.copy()

            y_train_raw_cur = y_train_raw.copy()
            y_test_raw_cur = y_test_raw.copy()

            y_train_raw_prev = None

            change = 0

            best_test_acc = -1
            self.ldc = None

            for iter_ in range(self.max_iter):

                # Actual IFN iterations start now
                X_train_feat = features.feat_ext(
                    X_train_raw_cur, self.features_)
                # X_test_feat = features.feat_ext(X_test_raw_cur, self.features_)

                ldc = discriminant_analysis.LinearDiscriminantAnalysis()
                ldc.fit(X_train_feat, y_train_raw)

                probs_ldc = ldc.predict_proba(X_train_feat)[
                    :, self.neutral_label]

                spkrs_neu_pred = dict()

                for i in range(len(probs_ldc)):
                    if probs_ldc[i] > self.neu_threshold:
                        if spkrs_train[i] not in spkrs_neu_pred:
                            spkrs_neu_pred[spkrs_train[i]] = 0
                        spkrs_neu_pred[spkrs_train[i]] += 1

                y_train_raw_prev = y_train_raw_cur.copy()

                # print(np.unique(spkrs_train, return_counts=True))

                y_train_raw_cur[probs_ldc >
                                self.neu_threshold] = self.neutral_label
                y_train_raw_cur[probs_ldc <=
                                self.neu_threshold] = self.emo_label

                for spkr in np.unique(spkrs_train):
                    if spkr not in spkrs_neu_pred:
                        spkrs_neu_pred[spkr] = 0
                    # print(spkr, len(spkrs_train[
                    #       spkrs_train == spkr]), spkrs_neu_pred[spkr])

                    if spkrs_neu_pred[spkr] / len(spkrs_train[spkrs_train == spkr]) < self.spkr_file_threshold:

                        # argsort_spkr_ll = np.argsort(-1 *
                        # probs_ldc[spkrs_train == spkr])

                        argsort_spkr_ll = sorted(np.where(spkrs_train == spkr)[
                                                 0].tolist(), key=lambda x: probs_ldc[x], reverse=True)

                        spkr_neu_num = int(
                            np.ceil(0.2 * len(spkrs_train[spkrs_train == spkr])))
                        neu_spkr_idx = argsort_spkr_ll[:spkr_neu_num]
                        emo_spkr_idx = argsort_spkr_ll[spkr_neu_num:]

                        # print(neu_spkr_idx, emo_spkr_idx)

                        y_train_raw_cur[neu_spkr_idx] = self.neutral_label
                        y_train_raw_cur[emo_spkr_idx] = self.emo_label

                    if self.norm_scheme == "ifn":

                        ref_avg_f0 = sum([sum(self.ref_corpus[i]) / len(self.ref_corpus[i])
                                          for i in range(len(self.ref_corpus))]) / len(self.ref_corpus)
                        # print(ref_avg_f0)

                        neu_spkr_idx = np.where((spkrs_train == spkr) & (
                            y_train_raw_cur == self.neutral_label))[0].tolist()

                        spkr_neu_f0 = [X_train_raw[i]
                                       for i in neu_spkr_idx]
                        # print(spkr_neu_f0)

                        neu_spkr_f0 = sum(
                            [sum(i) / len(i) for i in spkr_neu_f0]) / len(spkr_neu_f0)

                        self.norm_constants[
                            spkr] = ref_avg_f0 / neu_spkr_f0

                        for i in range(len(X_train_raw)):
                            if spkrs_train[i] == spkr:
                                X_train_raw_cur[i] = X_train_raw[
                                    i] / self.norm_constants[spkr]

                        for i in range(len(X_test_raw_cur)):
                            if spkrs_train[i] == spkr:
                                X_test_raw_cur[i] = X_test_raw[
                                    i] / self.norm_constants[spkr]

                        # print(self.norm_constants[spkr])

                    elif self.norm_scheme == "opt":

                        ref_avg_f0 = sum([sum(self.ref_corpus[i]) / len(self.ref_corpus[i])
                                          for i in range(len(self.ref_corpus))]) / len(self.ref_corpus)

                        neu_spkr_f0 = sum([sum(i) for i in X_train_raw[(spkrs_train == spkr) & (y_train_raw == self.neutral_label)]]
                                          / len(i)) / len(X_train_raw[(spkrs_train == spkr) & (y_train_raw == self.neutral_label)])

                        self.norm_constants[
                            spkr] = ref_avg_f0 / neu_spkr_f0

                        # X_train_raw_cur[spkrs_train == spkr] = X_train_raw[
                        #     spkr_train == spkr] / self.norm_constants[spkr]

                        X_train_raw_cur = [X_train_raw_cur[i] / self.norm_constants[spkr]
                                           for i in range(len(X_train_raw_cur)) if spkrs_train[i] == spkr]

                    elif self.norm_scheme == "none":

                        continue

                change = np.sum(y_train_raw_cur !=
                                y_train_raw_prev) / len(y_train_raw_prev)

                self.ldc = ldc

                clf_report = None

                if self.analysis:
                    X_test_feat = features.feat_ext(
                        X_test_raw_cur, self.features_)
                    ldc_prob_test = ldc.predict_proba(
                        X_test_feat)[:, self.neutral_label]

                    ldc_labels_test = np.int64(
                        ldc_prob_test > self.neu_threshold)
                    clf_report = classification_report(
                        y_test_raw, ldc_labels_test, output_dict=True)

                    print("ifn_iter_" + str(iter_))
                    print(classification_report(
                        y_test_raw, ldc_labels_test))
                    print("\n")
                    print("changed_files:", change)
                    print("\n\n")

                    if clf_report["accuracy"] > best_test_acc:
                        best_test_acc = clf_report["accuracy"]
                        self.best_ldc = ldc

                if change <= self.switch_threshold:
                    break

                else:

                    ss_iter_dir = self.log_dir + "ss_iter_" + str(ss) + "/"
                    if not os.path.exists(ss_iter_dir):
                        os.mkdir(ss_iter_dir)

                    ifn_iter_dir = ss_iter_dir + str(iter_) + "/"
                    if not os.path.exists(ifn_iter_dir):
                        os.mkdir(ifn_iter_dir)

                    with open(ifn_iter_dir + "clf_report.pkl", "wb") as f:
                        pickle.dump(clf_report, f)

                    with open(ifn_iter_dir + "ref_gmms.pkl", "wb") as f:
                        pickle.dump(self.ref_gmms, f)

                    with open(ifn_iter_dir + "ldc.pkl", "wb") as f:
                        pickle.dump(self.ldc, f)

                    with open(ifn_iter_dir + "best_ldc.pkl", "wb") as f:
                        pickle.dump(self.best_ldc, f)

                    with open(ifn_iter_dir + "change_prec.txt", "w") as f:
                        f.write(str(change))


def main():
    ref_file = "/Volumes/drive/data/RAVDESS_ref.pkl"

    with open(ref_file, "rb") as f:
        ref_corpus = pickle.load(f)

    ref_corpus = [i[1] for i in ref_corpus]

    features_ = list(features.feature_map.keys())

    ifn = IFN(ref_corpus, enable_neutral=True,
              feat=features_, ss_iter=1, norm_scheme="ifn", max_iter=100, neu_threshold=0.5)

    emo_file = "/Volumes/drive/data/RAVDESS_emo.pkl"

    with open(emo_file, "rb") as f:
        emo_corpus = pickle.load(f)

    X_emo = [i[1] for i in emo_corpus]
    spkrs = np.array([i[2] for i in emo_corpus])
    emo_labels = np.array([i[3] for i in emo_corpus])
    # print(emo_labels)
    # sys.exit()

    ifn.fit(X_emo, emo_labels, spkrs)


if __name__ == '__main__':
    main()
