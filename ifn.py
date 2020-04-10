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

from tqdm.auto import tqdm

import features


class IFN(object):

    def __init__(self, ref_corpus=None, enable_neutral=True, feat=["SQ75", "SQ25", "Smedian"], max_iter=10,
                 ss_iter=400, neutral_label=1, emo_label=0, analysis=True,
                 analysis_test_size=0.35, norm_scheme="ifn", neu_threshold=0.7,
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

            self._fit_ref()

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

        self.opt_norm_constants = dict()

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()

    def _fit_ref(self):

        X_ref = features.feat_ext(self.ref_corpus, self.features_)

        self.ref_feat = X_ref.copy()

        y_ref = np.array([self.neutral_label] * len(self.ref_corpus))

        for i in range(len(self.features_)):
            gmm = mixture.GaussianMixture(n_components=1)
            # print(len(np.unique(X_ref[:, i]).tolist()), X_ref[:, i].shape)
            gmm.fit(X_ref[:, i].reshape(X_ref.shape[0], 1), y_ref)

            self.ref_gmms.append(gmm)

    def fit(self, X_train_raw, y_train_raw, spkrs, save=None):

        # X_train_raw should be a list of pitch contours
        # y_train_raw should be np array of labels
        # spkrs should be an np array of speakers
        self.X_train_raw = X_train_raw.copy()
        self.y_train_raw = y_train_raw.copy()
        self.spkrs = spkrs.copy()

        num_neu_samples = len(y_train_raw[y_train_raw == self.neutral_label])
        num_emo_samples = len(y_train_raw[y_train_raw != self.neutral_label])

        diff = np.abs(num_emo_samples - num_neu_samples) / num_neu_samples

        if diff > 0.1:
            self.balance = True
        else:
            self.ss_iter = 1

        ss = 0
        for ss in tqdm(range(self.ss_iter), desc="ss_iter"):

            X_train_raw_ss = self.X_train_raw.copy()
            y_train_raw_ss = self.y_train_raw.copy()
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

                # sample population of emotional samples equal in size to
                # neutral samples for each speaker
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

            X_test_raw = None
            y_test_raw = None
            spkrs_test = None

            if self.analysis:
                X_train_raw, X_test_raw, y_train_raw, y_test_raw, spkrs_train, spkrs_test = train_test_split(X_train_raw_ss,
                                                                                                             y_train_raw_ss,
                                                                                                             spkrs_ss,
                                                                                                             test_size=self.analysis_test_size,
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

                if self.enable_neutral:

                    for i in range(len(self.ref_gmms)):
                        X_train_feat[:, i] = self.ref_gmms[
                            i].score_samples(X_train_feat[:, i].reshape(-1, 1))

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

                self.ldc = ldc

                clf_report = None

                if self.analysis:
                    X_test_feat = features.feat_ext(
                        X_test_raw_cur, self.features_)

                    if self.enable_neutral:
                        for i in range(len(self.ref_gmms)):
                            X_test_feat[:, i] = self.ref_gmms[i].score_samples(
                                X_test_feat[:, i].reshape(-1, 1))

                    ldc_prob_test = ldc.predict_proba(
                        X_test_feat)[:, self.neutral_label]

                    ldc_labels_test = np.int64(
                        ldc_prob_test > self.neu_threshold)
                    clf_report = classification_report(
                        y_test_raw, ldc_labels_test, output_dict=True, zero_division=0)

                    # print("ifn_iter_" + str(iter_))
                    # print(classification_report(
                    #     y_test_raw, ldc_labels_test))
                    # print("\n")

                    if clf_report["accuracy"] > best_test_acc:
                        best_test_acc = clf_report["accuracy"]
                        self.best_ldc = ldc

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

                        if spkr not in self.opt_norm_constants:
                            opt_neu_spkr_idx = np.where(
                                spkrs_train == spkr)[0].tolist()
                            opt_spkr_neu_f0 = [X_train_raw[i]
                                               for i in opt_neu_spkr_idx]
                            opt_neu_spkr_f0 = sum(
                                [sum(i) / len(i) for i in opt_spkr_neu_f0]) / len(opt_spkr_neu_f0)

                            self.opt_norm_constants[
                                spkr] = ref_avg_f0 / opt_neu_spkr_f0

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
                        # print(ref_avg_f0)

                        neu_spkr_idx = np.where((spkrs_train == spkr) & (
                            y_train_raw == self.neutral_label))[0].tolist()

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

                    elif self.norm_scheme == "global":

                        ref_avg_f0 = sum([sum(self.ref_corpus[i]) / len(self.ref_corpus[i])
                                          for i in range(len(self.ref_corpus))]) / len(self.ref_corpus)
                        # print(ref_avg_f0)

                        neu_spkr_idx = np.where(spkrs_train == spkr)[
                            0].tolist()

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

                    elif self.norm_scheme == "none":

                        continue

                change = np.sum(y_train_raw_cur !=
                                y_train_raw_prev) / len(y_train_raw_prev)

                if self.analysis:

                    # print("changed_files:", change)
                    # print("\n\n")

                    ss_iter_dir = self.log_dir + "ss_iter_" + str(ss) + "/"
                    if not os.path.exists(ss_iter_dir):
                        os.mkdir(ss_iter_dir)

                    # ifn_iter_dir = ss_iter_dir + \
                    #     "ifn_iter_" + str(iter_) + "/"
                    # if not os.path.exists(ifn_iter_dir):
                    #     os.mkdir(ifn_iter_dir)

                    ifn_iter_dir = ss_iter_dir + \
                        "ifn_iter_" + str(iter_) + "__"

                    # with open(ifn_iter_dir + "cur_model.pkl", "wb") as f:
                    #     pickle.dump(self.ldc, f)

                    # with open(ifn_iter_dir + "ref_models.pkl", "wb") as f:
                    #     pickle.dump(self.ref_gmms, f)

                    if ss == 0:
                        with open(self.log_dir + "feats.pkl", "wb") as f:
                            pickle.dump([i for i in features.feature_map if features.feature_map[
                                        i] in self.features_], f)

                    if iter_ == 0:
                        with open(ss_iter_dir + "opt_norm_constants.pkl", "wb") as f:
                            pickle.dump(self.opt_norm_constants, f)

                    with open(ifn_iter_dir + "norm_constants.pkl", "wb") as f:
                        pickle.dump(self.norm_constants, f)

                    with open(ifn_iter_dir + "clf_report.pkl", "wb") as f:
                        pickle.dump(clf_report, f)

                    with open(ifn_iter_dir + "change_perc.txt", "w") as f:
                        f.write(str(change))

                if iter_ == 0:
                    change = np.nan

                if iter_ != 0 and change <= self.switch_threshold:
                    break

                if self.norm_scheme != "ifn" and iter_ == 1:
                    # after one iteration in any non-IFN scheme, results will
                    # be same
                    break

            if self.analysis:
                with open(self.log_dir + "best_model.pkl", "wb") as f:
                    pickle.dump(self.best_ldc, f)

    def predict(self, X_test, spkrs_test, best=True, probs=False):

        for i in range(len(X_test)):
            X_test[i] = X_test[i] / self.norm_constants[spkrs_test[i]]

        X_test_feat = features.feat_ext(X_test, self.features)

        if self.enable_neutral:
            for i in range(len(self.features)):
                X_test_feat[:, i] = self.ref_gmms[i].score_samples(
                    X_test_feat[:, i].reshape(-1, 2))

        pred_prob = None
        if best:
            pred_prob = self.best_model.predict_proba(
                X_test_feat)[:, self.neutral_label]
        else:
            pred_prob = self.ldc.predict_proba(
                X_test_feat)[:, self.neutral_label]

        if probs:
            return pred_prob
        else:
            return pred_prob > self.neu_threshold

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Script for running Iterative Feature Normalization.')

    parser.add_argument('--ref_corpus', type=str,
                        help='dump of the reference corpus', required=True)

    parser.add_argument('--emo_corpus', type=str,
                        help='dump of the emotional corpus', required=True)

    parser.add_argument('--analysis', action="store_true",
                        help="if you want to generate detailed results for your run, similar to the paper.")

    parser.add_argument('--log_dir', type=str,
                        help='log directory where you want to dump all results', required=True)

    parser.add_argument('--enable_neutral', action='store_true',
                        help='follow "neutral" scheme for training')

    parser.add_argument('--feat', help="list of features to be used. See README for options.",
                        default=",".join(list(features.feature_map.keys())), type=str)

    parser.add_argument(
        '--max_iter', help="maximum no. of iterations for an IFN simulation", type=int, default=100)

    parser.add_argument(
        '--ss_iter', help="number of IFN simulations to perform. (only matters with --analysis flag)", type=int, default=1)

    parser.add_argument("--nlabel", type=int, default=1,
                        help="label for neutral instance")

    parser.add_argument("--elabel", type=int, default=0,
                        help="label for emotioal instance")

    parser.add_argument("--test_size", type=float, default=0.35,
                        help="test size at the time of analysis. (only matters with --analysis flag)")

    parser.add_argument("--norm_scheme", type=str, default="ifn",
                        help="normalization scheme ('ifn', 'none', 'opt', 'global')")

    parser.add_argument("--neu_threshold", type=float, default=0.45,
                        help="likelihood threshold for claddifying an instance as neutral")

    parser.add_argument("--spkr_file_threshold", type=float, default=0.2,
                        help="minimum number of files to be classified as neutral for each speaker")

    parser.add_argument("--switch_threshold", type=float, default=0.05,
                        help="minimum files to switch for IFN to continue")

    args = parser.parse_args()

    ref_file = args.ref_corpus

    with open(ref_file, "rb") as f:
        ref_corpus = pickle.load(f)

    ref_corpus = [i[1] for i in ref_corpus]

    features_ = args.feat.split(",")

    emo_file = args.emo_corpus

    with open(emo_file, "rb") as f:
        emo_corpus = pickle.load(f)

    X_emo = [i[1] for i in emo_corpus]
    spkrs = np.array([i[2] for i in emo_corpus])
    emo_labels = np.array([i[3] for i in emo_corpus])

    ifn = IFN(ref_corpus=ref_corpus, enable_neutral=args.enable_neutral, feat=features_, max_iter=args.max_iter,
              ss_iter=args.ss_iter, neutral_label=args.nlabel, emo_label=args.elabel, analysis=args.analysis,
              analysis_test_size=args.test_size, norm_scheme=args.norm_scheme, neu_threshold=args.neu_threshold,
              spkr_file_threshold=args.spkr_file_threshold, switch_threshold=args.switch_threshold, log_dir=args.log_dir)

    ifn.fit(X_emo, emo_labels, spkrs)
