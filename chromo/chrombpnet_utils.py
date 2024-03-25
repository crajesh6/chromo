from typing import Any
import json
import math
import os
import random as rn

import h5py
import matplotlib

matplotlib.use("Agg")  # Set the 'Agg' backend before importing pyplot
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import interpn
from scipy.spatial.distance import jensenshannon
from scipy.stats import multinomial, spearmanr, pearsonr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import (
    Input,
    Cropping1D,
    Conv1D,
    GlobalAvgPool1D,
    Dense,
    Add,
    Concatenate,
    Lambda,
    Flatten,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_custom_objects
import tensorflow_probability as tfp
from tqdm import tqdm

import pyBigWig
import pyfaidx
import polars as pl

from chromo import layers, model_zoo, utils

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]


def take_per_row(A, indx, num_elem):
    """
    Matrix A, indx is a vector for each row which specifies
    slice beginning for that row. Each has width num_elem.
    """

    all_indx = indx[:, None] + np.arange(num_elem)
    return A[np.arange(all_indx.shape[0])[:, None], all_indx]


def random_crop(seqs, labels, seq_crop_width, label_crop_width, coords):
    """
    Takes sequences and corresponding counts labels. They should have the same
    #examples. The widths would correspond to inputlen and outputlen respectively,
    and any additional flanking width for jittering which should be the same
    for seqs and labels. Each example is cropped starting at a random offset.

    seq_crop_width - label_crop_width should be equal to seqs width - labels width,
    essentially implying they should have the same flanking width.
    """

    assert seqs.shape[1] >= seq_crop_width
    assert labels.shape[1] >= label_crop_width
    assert seqs.shape[1] - seq_crop_width == labels.shape[1] - label_crop_width

    max_start = (
        seqs.shape[1] - seq_crop_width
    )  # This should be the same for both input and output

    starts = np.random.choice(range(max_start + 1), size=seqs.shape[0], replace=True)

    new_coords = coords.copy()
    new_coords[:, 1] = new_coords[:, 1].astype(int) - (seqs.shape[1] // 2) + starts

    return (
        take_per_row(seqs, starts, seq_crop_width),
        take_per_row(labels, starts, label_crop_width),
        new_coords,
    )


def random_rev_comp(seqs, labels, coords, frac=0.5):
    """
    Data augmentation: applies reverse complement randomly to a fraction of
    sequences and labels.

    Assumes seqs are arranged in ACGT. Then ::-1 gives TGCA which is revcomp.

    NOTE: Performs in-place modification.
    """

    pos_to_rc = np.random.choice(
        range(seqs.shape[0]), size=int(seqs.shape[0] * frac), replace=False
    )

    seqs[pos_to_rc] = seqs[pos_to_rc, ::-1, ::-1]
    labels[pos_to_rc] = labels[pos_to_rc, ::-1]
    coords[pos_to_rc, 2] = "r"

    return seqs, labels, coords


def crop_revcomp_augment(
    seqs,
    labels,
    coords,
    seq_crop_width,
    label_crop_width,
    add_revcomp,
    rc_frac=0.5,
    shuffle=False,
):
    """
    seqs: B x IL x 4
    labels: B x OL

    Applies random crop to seqs and labels and reverse complements rc_frac.
    """

    assert seqs.shape[0] == labels.shape[0]

    # this does not modify seqs and labels
    # mod_seqs, mod_labels, mod_coords = random_crop(seqs, labels, seq_crop_width, label_crop_width, coords)
    mod_seqs, mod_labels, mod_coords = seqs, labels, coords

    # this modifies mod_seqs, mod_labels in-place
    if add_revcomp:
        mod_seqs, mod_labels, mod_coords = random_rev_comp(
            mod_seqs, mod_labels, mod_coords, frac=rc_frac
        )

    if shuffle:
        perm = np.random.permutation(mod_seqs.shape[0])
        mod_seqs = mod_seqs[perm]
        mod_labels = mod_labels[perm]
        mod_coords = mod_coords[perm]

    return mod_seqs, mod_labels, mod_coords


def one_hot_encode(
    sequence: str,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: Any = 0,
    dtype=np.uint8,
) -> np.ndarray:
    """One-hot encode sequence."""

    def to_uint8(s):
        return np.frombuffer(s.encode("ascii"), dtype=np.uint8)

    lookup = np.zeros([np.iinfo(np.uint8).max, len(alphabet)], dtype=dtype)
    lookup[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    lookup[to_uint8(neutral_alphabet)] = neutral_value
    lookup = lookup.astype(dtype)
    return lookup[to_uint8(sequence)]


def one_hot_to_dna(one_hot):
    """
    Converts a one-hot encoding into a list of DNA ("ACGT") sequences, where the
    position of 1s is ordered alphabetically by "ACGT". `one_hot` must be an
    N x L x 4 array of one-hot encodings. Returns a lits of N "ACGT" strings,
    each of length L, in the same order as the input array. The returned
    sequences will only consist of letters "A", "C", "G", "T", or "N" (all
    upper-case). Any encodings that are all 0s will be translated to "N".
    """
    bases = np.array(["A", "C", "G", "T", "N"])
    # Create N x L array of all 5s
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

    # Get indices of where the 1s are
    batch_inds, seq_inds, base_inds = np.where(one_hot)

    # In each of the locations in the N x L array, fill in the location of the 1
    one_hot_inds[batch_inds, seq_inds] = base_inds

    # Fetch the corresponding base for each position using indexing
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]


def get_seq(peaks_df, genome, width):
    """
    Same as get_cts, but fetches sequence from a given genome.
    """
    vals = []
    print("Getting seqs!")

    for i in tqdm(peaks_df.iter_rows(), total=len(peaks_df)):
        chr, start, end = i[:3]
        summit = i[-1]
        sequence = str(
            genome[chr][(start + summit - width // 2) : (start + summit + width // 2)]
        )
        vals += [sequence]

    seqs = []
    for _, v in tqdm(enumerate(vals), total=len(vals)):
        seqs += [one_hot_encode(v)]

    seqs = np.stack(seqs)

    return seqs


def get_cts(peaks_df, bw, width):
    """
    Fetches values from a bigwig bw, given a df with minimally
    chr, start and summit columns. Summit is relative to start.
    Retrieves values of specified width centered at summit.

    "cts" = per base counts across a region
    """
    vals = []
    print("Getting counts!")
    for i in tqdm(peaks_df.iter_rows(), total=len(peaks_df)):
        chr, start, end = i[:3]
        summit = i[-1]
        vals.append(
            np.nan_to_num(
                bw.values(chr, start + summit - width // 2, start + summit + width // 2)
            )
        )

    return np.array(vals)


def get_coords(peaks_df, peaks_bool):
    """
    Fetch the co-ordinates of the regions in bed file
    returns a list of tuples with (chrom, summit)
    """
    vals = []
    print("Getting coords!")
    for i in tqdm(peaks_df.iter_rows(), total=len(peaks_df)):
        chr, start, end = i[:3]
        summit = i[-1]
        vals.append([chr, start + summit, "f", peaks_bool])

    return np.array(vals)


def _get_coords(peaks_df, peaks_bool):
    """
    Fetch the co-ordinates of the regions in bed file
    returns a list of tuples with (chrom, summit)
    """
    vals = []
    full_vals = []
    for i, r in peaks_df.iterrows():
        vals.append([r["chr"], r["start"] + r["summit"], "f", peaks_bool])
        full_vals.append([r["chr"], r["start"], r["end"], r["summit"], "f", peaks_bool])

    return np.array(vals), np.array(full_vals)


def get_seq_cts_coords(peaks_df, genome, bw, input_width, output_width, peaks_bool):
    seq = get_seq(peaks_df, genome, input_width)
    cts = get_cts(peaks_df, bw, output_width)
    coords = get_coords(peaks_df, peaks_bool)
    return seq, cts, coords


def load_data(
    bed_regions,
    nonpeak_regions,
    genome_fasta,
    cts_bw_file,
    inputlen,
    outputlen,
    max_jitter,
    smoke_test,
):
    """
    Load sequences and corresponding base resolution counts for training,
    validation regions in peaks and nonpeaks (2 x 2 x 2 = 8 matrices).

    For training peaks/nonpeaks, values for inputlen + 2*max_jitter and outputlen + 2*max_jitter
    are returned centered at peak summit. This allows for jittering examples by randomly
    cropping. Data of width inputlen/outputlen is returned for validation
    data.

    If outliers is not None, removes training examples with counts > outlier%ile
    """

    cts_bw = pyBigWig.open(cts_bw_file)
    genome = pyfaidx.Fasta(genome_fasta)

    train_peaks_seqs = None
    train_peaks_cts = None
    train_peaks_coords = None
    train_nonpeaks_seqs = None
    train_nonpeaks_cts = None
    train_nonpeaks_coords = None

    if bed_regions is not None:
        train_peaks_seqs, train_peaks_cts, train_peaks_coords = get_seq_cts_coords(
            bed_regions,
            genome,
            cts_bw,
            inputlen + 2 * max_jitter,
            outputlen + 2 * max_jitter,
            peaks_bool=1,
        )

    if nonpeak_regions is not None:
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords = (
            get_seq_cts_coords(
                nonpeak_regions, genome, cts_bw, inputlen, outputlen, peaks_bool=0
            )
        )

    cts_bw.close()
    genome.close()

    return (
        train_peaks_seqs,
        train_peaks_cts,
        train_peaks_coords,
        train_nonpeaks_seqs,
        train_nonpeaks_cts,
        train_nonpeaks_coords,
    )


def subsample_nonpeak_data(
    nonpeak_seqs, nonpeak_cts, nonpeak_coords, peak_data_size, negative_sampling_ratio
):
    # Randomly samples a portion of the non-peak data to use in training
    num_nonpeak_samples = int(negative_sampling_ratio * peak_data_size)
    nonpeak_indices_to_keep = np.random.choice(
        len(nonpeak_seqs), size=num_nonpeak_samples, replace=False
    )
    nonpeak_seqs = nonpeak_seqs[nonpeak_indices_to_keep]
    nonpeak_cts = nonpeak_cts[nonpeak_indices_to_keep]
    nonpeak_coords = nonpeak_coords[nonpeak_indices_to_keep]
    return nonpeak_seqs, nonpeak_cts, nonpeak_coords


class ChromBPNetBatchGenerator(keras.utils.Sequence):
    """
    This generator randomly crops (=jitter) and revcomps training examples for
    every epoch, and calls bias model on it, whose outputs (bias profile logits
    and bias logcounts) are fed as input to the chrombpnet model.
    """

    def __init__(
        self,
        peak_regions,
        nonpeak_regions,
        genome_fasta,
        batch_size,
        inputlen,
        outputlen,
        max_jitter,
        negative_sampling_ratio,
        cts_bw_file,
        add_revcomp,
        return_coords,
        shuffle_at_epoch_start,
        smoke_test,
        seq_only=False,
    ):
        """
        seqs: B x L' x 4
        cts: B x M'
        inputlen: int (L <= L'), L' is greater to allow for cropping (= jittering)
        outputlen: int (M <= M'), M' is greater to allow for cropping (= jittering)
        batch_size: int (B)
        """
        self.smoke_test = smoke_test
        (
            peak_seqs,
            peak_cts,
            peak_coords,
            nonpeak_seqs,
            nonpeak_cts,
            nonpeak_coords,
        ) = load_data(
            peak_regions,
            nonpeak_regions,
            genome_fasta,
            cts_bw_file,
            inputlen,
            outputlen,
            max_jitter,
            self.smoke_test,
        )
        self.peak_seqs, self.nonpeak_seqs = peak_seqs, nonpeak_seqs
        self.peak_cts, self.nonpeak_cts = peak_cts, nonpeak_cts
        self.peak_coords, self.nonpeak_coords = peak_coords, nonpeak_coords

        self.negative_sampling_ratio = negative_sampling_ratio
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.batch_size = batch_size
        self.add_revcomp = add_revcomp
        self.return_coords = return_coords
        self.seq_only = seq_only
        self.shuffle_at_epoch_start = shuffle_at_epoch_start

        # random crop training data to the desired sizes, revcomp augmentation
        self.crop_revcomp_data()

    def __len__(self):
        return math.ceil(self.seqs.shape[0] / self.batch_size)

    def crop_revcomp_data(self):
        # random crop training data to inputlen and outputlen (with corresponding offsets), revcomp augmentation
        # shuffle required since otherwise peaks and nonpeaks will be together
        # Sample a fraction of the negative samples according to the specified ratio
        if (self.peak_seqs is not None) and (self.nonpeak_seqs is not None):
            # crop peak data before stacking
            cropped_peaks, cropped_cnts, cropped_coords = random_crop(
                self.peak_seqs,
                self.peak_cts,
                self.inputlen,
                self.outputlen,
                self.peak_coords,
            )
            # print(cropped_peaks.shape)
            # print(self.nonpeak_seqs.shape)
            if self.negative_sampling_ratio < 1.0:
                (
                    self.sampled_nonpeak_seqs,
                    self.sampled_nonpeak_cts,
                    self.sampled_nonpeak_coords,
                ) = subsample_nonpeak_data(
                    self.nonpeak_seqs,
                    self.nonpeak_cts,
                    self.nonpeak_coords,
                    len(self.peak_seqs),
                    self.negative_sampling_ratio,
                )
                self.seqs = np.vstack([cropped_peaks, self.sampled_nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, self.sampled_nonpeak_cts])
                self.coords = np.vstack([cropped_coords, self.sampled_nonpeak_coords])
            else:
                self.seqs = np.vstack([cropped_peaks, self.nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, self.nonpeak_cts])
                self.coords = np.vstack([cropped_coords, self.nonpeak_coords])

        elif self.peak_seqs is not None:
            # crop peak data before stacking
            cropped_peaks, cropped_cnts, cropped_coords = random_crop(
                self.peak_seqs,
                self.peak_cts,
                self.inputlen,
                self.outputlen,
                self.peak_coords,
            )

            self.seqs = cropped_peaks
            self.cts = cropped_cnts
            self.coords = cropped_coords

        elif self.nonpeak_seqs is not None:
            # print(self.nonpeak_seqs.shape)

            self.seqs = self.nonpeak_seqs
            self.cts = self.nonpeak_cts
            self.coords = self.nonpeak_coords
        else:
            print("Both peak and non-peak arrays are empty")

        self.cur_seqs, self.cur_cts, self.cur_coords = crop_revcomp_augment(
            self.seqs,
            self.cts,
            self.coords,
            self.inputlen,
            self.outputlen,
            self.add_revcomp,
            shuffle=self.shuffle_at_epoch_start,
        )

    def __getitem__(self, idx):
        batch_seq = self.cur_seqs[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_cts = self.cur_cts[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_coords = self.cur_coords[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        if self.seq_only:
            return batch_seq

        if self.return_coords:
            return (
                batch_seq,
                [batch_cts, np.log(1 + batch_cts.sum(-1, keepdims=True))],
                batch_coords,
            )
        else:
            return (
                batch_seq,
                [batch_cts, np.log(1 + batch_cts.sum(-1, keepdims=True))],
            )

    def on_epoch_end(self):
        self.crop_revcomp_data()


def fetch_data_and_model_params_based_on_mode(
    mode,
    nonpeak_regions,
    peak_regions,
    seed=1234,
    inputlen=2114,
    outputlen=1000,
    negative_sampling_ratio=0.1,
    max_jitter=500,
):
    if mode == "train":
        add_revcomp = True
        shuffle_at_epoch_start = True

    elif mode == "valid":
        # fix negatives set for validation
        if (nonpeak_regions is not None) and (peak_regions is not None):
            nonpeak_regions = nonpeak_regions.sample(
                n=int(float(negative_sampling_ratio) * peak_regions.shape[0]),
                with_replacement=False,
                seed=seed,
            )
        negative_sampling_ratio = 1.0  # already subsampled
        # do not jitter at valid time - we are testing only at summits
        max_jitter = 0
        # no reverse complementation at valid time
        add_revcomp = False
        # no need to shuffle
        shuffle_at_epoch_start = False

    elif mode == "test":
        # no subsampling of negatives - test on all positives and negatives
        negative_sampling_ratio = 1.0
        # no jitter at valid time - we are testing only at summits
        max_jitter = 0
        # no reverse complementation at test time
        add_revcomp = False
        # no need to shuffle
        shuffle_at_epoch_start = False

    else:
        print("mode not defined - only train, valid, test are allowed")

    return (
        inputlen,
        outputlen,
        nonpeak_regions,
        negative_sampling_ratio,
        max_jitter,
        add_revcomp,
        shuffle_at_epoch_start,
    )


def get_bed_regions_for_fold_split(bed_regions, mode, splits_dict):
    chroms_to_keep = splits_dict[mode]
    # bed_regions_to_keep=bed_regions[bed_regions["chr"].isin(chroms_to_keep)]
    bed_regions_to_keep = bed_regions.filter(pl.col("chr").is_in(chroms_to_keep))
    print(
        "got split:" + str(mode) + " for bed regions:" + str(bed_regions_to_keep.shape)
    )
    return bed_regions_to_keep, chroms_to_keep


def initialize_generators(
    mode,
    chr_fold_path="/home/chandana/projects/chromo/data/fold_0.json",
    peaks="/home/chandana/projects/chromo/data/filtered.peaks.bed",
    nonpeaks="/home/chandana/projects/chromo/data/filtered.nonpeaks.bed",
    genome="/home/chandana/projects/chromo/data/hg38.fa",
    bigwig="/home/chandana/projects/chromo/data/merged_unstranded.bw",
    seed=1234,
    batch_size=64,
    inputlen=2114,
    outputlen=1000,
    negative_sampling_ratio=0.1,
    max_jitter=500,
    return_coords=False,
    smoke_test=False,
    seq_only=False,
):
    # defaults
    peak_regions = None
    nonpeak_regions = None

    # get only those peak/non peak regions corresponding to train/valid/test set
    splits_dict = json.load(open(chr_fold_path))

    if peaks.lower() != "none":
        print("loading peaks...")
        peak_regions = pl.read_csv(
            peaks, has_header=False, separator="\t", new_columns=NARROWPEAK_SCHEMA
        )
        if smoke_test:
            peak_regions = peak_regions.sample(n=10000, seed=0)
        peak_regions, chroms = get_bed_regions_for_fold_split(
            peak_regions, mode, splits_dict
        )

    if nonpeaks.lower() != "none":
        print("loading nonpeaks...")
        nonpeak_regions = pl.read_csv(
            nonpeaks, has_header=False, separator="\t", new_columns=NARROWPEAK_SCHEMA
        )
        if smoke_test:
            nonpeak_regions = nonpeak_regions.sample(n=10000, seed=0)
        nonpeak_regions, chroms = get_bed_regions_for_fold_split(
            nonpeak_regions, mode, splits_dict
        )

    (
        inputlen,
        outputlen,
        nonpeak_regions,
        negative_sampling_ratio,
        max_jitter,
        add_revcomp,
        shuffle_at_epoch_start,
    ) = fetch_data_and_model_params_based_on_mode(
        mode,
        nonpeak_regions,
        peak_regions,
        seed=seed,
        inputlen=inputlen,
        outputlen=outputlen,
        negative_sampling_ratio=negative_sampling_ratio,
        max_jitter=max_jitter,
    )

    generator = ChromBPNetBatchGenerator(
        peak_regions=peak_regions,
        nonpeak_regions=nonpeak_regions,
        genome_fasta=genome,
        batch_size=batch_size,
        inputlen=inputlen,
        outputlen=outputlen,
        max_jitter=max_jitter,
        negative_sampling_ratio=negative_sampling_ratio,
        cts_bw_file=bigwig,
        add_revcomp=add_revcomp,
        return_coords=return_coords,
        shuffle_at_epoch_start=shuffle_at_epoch_start,
        smoke_test=smoke_test,
        seq_only=seq_only,
    )

    return generator


################################################
# Model Building Code
################################################


# Set a fixed seed for reproducibility across multiple runs
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
tf.random.set_seed(42)
rn.seed(42)


def load_model_wrapper(model_h5):
    # read .h5 model
    custom_objects = {"tf": tf, "multinomial_nll": multinomial_nll}
    get_custom_objects().update(custom_objects)
    model = load_model(model_h5)
    print("got the model")
    model.summary()
    return model


def load_pretrained_bias(model_hdf5):
    custom_objects = {"multinomial_nll": multinomial_nll, "tf": tf}
    get_custom_objects().update(custom_objects)
    pretrained_bias_model = load_model(model_hdf5)
    # freeze the model
    num_layers = len(pretrained_bias_model.layers)
    for i in range(num_layers):
        pretrained_bias_model.layers[i].trainable = False
    return pretrained_bias_model


def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood."""
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example, logits=logits)
    return -tf.reduce_sum(dist.log_prob(true_counts)) / tf.cast(
        tf.shape(true_counts)[0], dtype=tf.float32
    )


def adjust_bias_model_logcounts(bias_model, seqs, cts):
    """
    Given a bias model, sequences and associated counts, the function adds a
    constant to the output of the bias_model's logcounts that minimises squared
    error between predicted logcounts and observed logcounts (infered from
    cts). This simply reduces to adding the average difference between observed
    and predicted to the "bias" (constant additive term) of the Dense layer.
    Typically the seqs and counts would correspond to training nonpeak regions.
    ASSUMES model_bias's last layer is a dense layer that outputs logcounts.
    This would change if you change the model.
    """

    # safeguards to prevent misuse
    # assert(bias_model.layers[-1].name == "logcount_predictions")
    # assert(bias_model.layers[-1].name == "logcounts" or bias_model.layers[-1].name == "logcount_predictions")
    assert bias_model.layers[-1].output_shape == (None, 1)
    assert isinstance(bias_model.layers[-1], keras.layers.Dense)

    print("Predicting within adjust counts")
    _, pred_logcts = bias_model.predict(seqs, verbose=True)
    delta = np.mean(np.log(1 + cts) - pred_logcts.ravel())

    dw, db = bias_model.layers[-1].get_weights()
    bias_model.layers[-1].set_weights([dw, db + delta])
    return bias_model


def bpnet_model(
    conv1_kernel_size=21,
    profile_kernel_size=75,
    num_tasks=1,
    filters=512,
    n_dilation_layers=8,
    counts_loss_weight=75.9,
    sequence_len=2114,
    out_pred_len=1000,
    name_prefix="bias",
    learning_rate=0.001,
):
    """Constructs and compiles the BPNet model based on provided parameters."""

    inp = Input(shape=(sequence_len, 4), name=f"{name_prefix}_sequence")
    x = Conv1D(
        filters,
        kernel_size=conv1_kernel_size,
        padding="valid",
        activation="relu",
        name=f"{name_prefix}_1st_conv",
    )(inp)

    for i in range(1, n_dilation_layers + 1):
        conv_layer_name = f"{name_prefix}_{i}conv"
        conv_x = Conv1D(
            filters,
            kernel_size=3,
            padding="valid",
            activation="relu",
            dilation_rate=2**i,
            name=conv_layer_name,
        )(x)
        x_len, conv_x_len = int_shape(x)[1], int_shape(conv_x)[1]
        x = Cropping1D((x_len - conv_x_len) // 2, name=f"{name_prefix}_{i}crop")(x)
        x = Add(name=f"{name_prefix}_{i}add")([conv_x, x])

    prof_out_precrop = Conv1D(
        filters=num_tasks,
        kernel_size=profile_kernel_size,
        padding="valid",
        name=f"{name_prefix}_prof_out_precrop",
    )(x)
    crop_size = (int_shape(prof_out_precrop)[1] // 2) - (out_pred_len // 2)
    prof = Cropping1D(
        crop_size, name=f"{name_prefix}_logits_profile_predictions_preflatten"
    )(prof_out_precrop)
    profile_out = Flatten(name=f"{name_prefix}_logits_profile_predictions")(prof)
    gap_combined_conv = GlobalAvgPool1D(name=f"{name_prefix}_gap")(x)
    count_out = Dense(num_tasks, name=f"{name_prefix}_logcount_predictions")(
        gap_combined_conv
    )

    model = Model(
        inputs=[inp], outputs=[profile_out, count_out], name=f"{name_prefix}_model"
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=[multinomial_nll, "mse"],
        loss_weights=[1, counts_loss_weight],
    )

    return model


def chrombpnet(
    seed=1234,
    filters=512,
    n_dilation_layers=8,
    counts_loss_weight=75.9,
    sequence_len=2114,
    out_pred_len=1000,
    bias_model_path="/home/chandana/projects/chrombpnet_pipeline/results/run_1/split_0/full_model/bias_model_scaled.h5",
    name_prefix="chrombpnet",
    learning_rate=0.001,
    **kwargs,
):
    """
    Constructs and compiles the combined model with bias and without bias outputs.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    bias_model = load_pretrained_bias(bias_model_path)
    bpnet_model_without_bias = bpnet_model(
        filters=filters,
        n_dilation_layers=n_dilation_layers,
        counts_loss_weight=counts_loss_weight,
        sequence_len=sequence_len,
        out_pred_len=out_pred_len,
        name_prefix=name_prefix,
        learning_rate=learning_rate,
    )

    inp = Input(shape=(sequence_len, 4), name=f"{name_prefix}_sequence")
    bias_output = bias_model(inp)
    output_without_bias = bpnet_model_without_bias(inp)

    assert (
        len(bias_output[1].shape) == 2
    ), "Bias model counts head is of incorrect shape (None,1) expected."
    assert (
        bias_output[1].shape[1] == 1
    ), "Bias model counts head is of incorrect shape (None,1) expected."
    assert (
        bias_output[0].shape[1] == out_pred_len
    ), "Bias model profile head is of incorrect shape (None,out_pred_len) expected."

    profile_out = Add(name=f"{name_prefix}_logits_profile_predictions")(
        [output_without_bias[0], bias_output[0]]
    )
    concat_counts = Concatenate(axis=-1)([output_without_bias[1], bias_output[1]])
    count_out = Lambda(
        lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_logcount_predictions",
    )(concat_counts)

    full_model = Model(inputs=[inp], outputs=[profile_out, count_out])
    full_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=[multinomial_nll, "mse"],
        loss_weights=[1, counts_loss_weight],
    )

    return full_model


def homininn_model(
    base_model,
    conv1_activation,
    conv1_batchnorm,
    conv1_dropout,
    conv1_filters,
    conv1_kernel_size,
    conv1_pool_size,
    conv_layer_type,
    decode_activation,
    decode_batchnorm,
    decode_dropout,
    decode_filters,
    decode_kernel_size,
    dense_activation,
    dense_batchnorm,
    dense_dropout,
    dense_units,
    downsample_factor,
    input_shape,
    mha_d_model,
    mha_dropout,
    mha_head_type,
    mha_heads,
    mha_layernorm,
    motif_pooling_type,
    output_activation,
    output_shape,
    num_tasks,
    spatial_pooling_type,
    positional_encoding="enformer",
):
    crop_size = (input_shape[0] - 2000) // 2
    name_prefix = "homininn_model"
    num_resid = 3
    task_filters = 64
    task_kernel_size = 5
    task_dropout = 0.2
    # task_activation = 'softplus'
    diag = l2(1e-6)
    offdiag = l2(1e-3)

    # zero-pad to ensure L can downsample exactly with 2^downsample
    max_pool = 2**downsample_factor

    inputs = keras.layers.Input(shape=input_shape)
    # crop input to be 2k
    nn = Cropping1D(crop_size, name=f"{name_prefix}_pre-conv_crop")(inputs)

    # convolutional layer
    print("hello!")
    # print(conv_layer_type, conv1_filters, conv1_kernel_size, diag, offdiag)
    print(
        model_zoo.create_conv_layer(
            conv_layer_type, conv1_filters, conv1_kernel_size, diag, offdiag
        )
    )
    nn = model_zoo.create_conv_layer(
        conv_layer_type, conv1_filters, conv1_kernel_size, diag, offdiag
    )(nn)
    if conv1_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn_connect = keras.layers.Activation(conv1_activation, name="conv_activation")(nn)
    nn = keras.layers.Dropout(conv1_dropout)(nn_connect)

    # choose motif pooling layer!
    nn = model_zoo.create_motif_pooling_layer(motif_pooling_type, nn, conv1_filters)

    # choose spatial pooling layer!
    nn = model_zoo.create_pooling_layer(
        spatial_pooling_type, max_pool, name="conv1_pool"
    )(nn)

    print(f"Shape of nn_connect: {nn_connect.shape}")

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    # Need to choose which positional encoding to use!
    if positional_encoding == "enformer":
        nn, att = layers.MultiHeadAttention(num_heads=mha_heads, d_model=mha_d_model)(
            nn, nn, nn
        )  # dims
    elif positional_encoding == "rope":  # TODO
        nn, att = layers.MultiHeadAttentionRoPE(
            num_heads=mha_heads, d_model=mha_d_model
        )(nn, nn, nn)  # dims
    nn = keras.layers.Dropout(mha_dropout)(nn)

    # expand back to base-resolution
    for i in range(downsample_factor):
        nn = keras.layers.Conv1DTranspose(
            filters=decode_filters,
            kernel_size=decode_kernel_size,
            strides=2,
            padding="same",
        )(nn)
        if decode_batchnorm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(decode_activation)(nn)
        nn = keras.layers.Dropout(decode_dropout)(nn)

    nn = keras.layers.Concatenate(axis=2)([nn, nn_connect])
    nn = keras.layers.Conv1D(filters=decode_filters, kernel_size=5, padding="same")(nn)
    if decode_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(decode_activation)(nn)
    nn = keras.layers.Dropout(decode_dropout)(nn)
    nn2 = residual_block(nn, 3, activation=decode_activation, dilated=num_resid)  # 3
    nn = keras.layers.add([nn, nn2])

    nn2 = keras.layers.Conv1D(
        filters=task_filters, kernel_size=task_kernel_size, padding="same"
    )(nn)
    nn2 = keras.layers.Activation(decode_activation)(nn2)
    nn2 = keras.layers.Dropout(task_dropout)(nn2)

    # chrombpnet specific output head
    # profile branch
    prof_out_precrop = Conv1D(
        filters=num_tasks,
        kernel_size=75,
        padding="same",
        name=f"{name_prefix}_profile_out",
    )(nn2)
    profile_out = Flatten(name=f"{name_prefix}_logits_profile_predictions")(
        prof_out_precrop
    )
    profile_out_1k = Dense(1000, name=f"{name_prefix}_logcount_predictions_1k")(
        profile_out
    )

    # counts branch
    gap_combined_conv = GlobalAvgPool1D(name=f"{name_prefix}_global_average_pool")(nn2)
    count_out = Dense(num_tasks, name=f"{name_prefix}_logcount_predictions")(
        gap_combined_conv
    )

    model = tf.keras.Model(
        inputs=[inputs], outputs=[profile_out_1k, count_out], name="homininn_model"
    )

    return model


def residual_block(input_layer, filter_size, activation="relu", dilated=5):
    factor = []
    base = 2
    for i in range(dilated):
        factor.append(base**i)

    num_filters = input_layer.shape.as_list()[-1]

    nn = keras.layers.Conv1D(
        filters=num_filters,
        kernel_size=filter_size,
        activation=None,
        use_bias=False,
        padding="same",
        dilation_rate=1,
    )(input_layer)
    nn = keras.layers.BatchNormalization()(nn)
    for f in factor:
        nn = keras.layers.Activation("relu")(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        nn = keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            strides=1,
            activation=None,
            use_bias=False,
            padding="same",
            dilation_rate=f,
        )(nn)
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)


def _factorized_homininn(args, model_params):
    """
    Constructs and compiles the combined model with bias and without bias outputs.
    """
    # Validate required parameters
    assert "bias_model_path" in model_params, "Bias model path not specified for model."

    # Initialize seeds for reproducibility
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Model parameters
    sequence_len = int(model_params["inputlen"])
    out_pred_len = int(model_params["outputlen"])
    counts_loss_weight = float(model_params["counts_loss_weight"])
    bias_model_path = model_params["bias_model_path"]
    name_prefix = "factorized_homininn"

    # Load models
    # Use the same bias model that was trained for chrombpnet
    bias_model = load_pretrained_bias(bias_model_path)
    config_path = args["config_path"]
    config = utils.load_config(config_path)
    model_wo_bias = homininn_model(**config)  # this might need to change!
    model_wo_bias.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=[multinomial_nll, "mse"],
        loss_weights=[1, counts_loss_weight],
    )

    # Define input
    inp = Input(shape=(sequence_len, 4), name=f"{name_prefix}_sequence")

    # Get model outputs
    bias_output = bias_model(inp)
    output_wo_bias = model_wo_bias(inp)

    # Validate output shapes
    assert (
        len(bias_output[1].shape) == 2
    ), "Bias model counts head is of incorrect shape (None,1) expected."
    assert (
        len(bias_output[0].shape) == 2
    ), "Bias model profile head is of incorrect shape (None,out_pred_len) expected."
    assert (
        bias_output[1].shape[1] == 1
    ), "Bias model counts head is of incorrect shape (None,1) expected."
    assert (
        bias_output[0].shape[1] == out_pred_len
    ), "Bias model profile head is of incorrect shape (None,out_pred_len) expected."

    # Combine outputs
    profile_out = Add(name=f"{name_prefix}_logits_profile_predictions")(
        [output_wo_bias[0], bias_output[0]]
    )
    concat_counts = Concatenate(axis=-1)([output_wo_bias[1], bias_output[1]])
    count_out = Lambda(
        lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_logcount_predictions",
    )(concat_counts)

    # Instantiate and compile the model
    full_model = Model(
        inputs=[inp], outputs=[profile_out, count_out], name="full_model"
    )
    full_model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=[multinomial_nll, "mse"],
        loss_weights=[1, counts_loss_weight],
    )

    return full_model


def factorized_homininn(
    seed=1234,
    counts_loss_weight=75.9,
    bias_model_path="/home/chandana/projects/chrombpnet_pipeline/results/run_1/split_0/full_model/bias_model_scaled.h5",
    name_prefix="factorized_homininn",
    learning_rate=0.001,
    **kwargs,
):
    """
    Constructs and compiles the combined model with bias and without bias outputs.
    """
    # Initialize seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load models
    bias_model = load_pretrained_bias(bias_model_path)

    # Use kwargs to configure the model without bias
    model_wo_bias = homininn_model(**kwargs)
    model_wo_bias.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=[multinomial_nll, "mse"],
        loss_weights=[1, counts_loss_weight],
    )

    # Define input
    inp = Input(shape=(kwargs["input_shape"][0], 4), name=f"{name_prefix}_sequence")

    # Get model outputs
    bias_output = bias_model(inp)
    output_wo_bias = model_wo_bias(inp)

    # Validate output shapes
    assert (
        len(bias_output[1].shape) == 2
    ), "Bias model counts head is of incorrect shape (None,1) expected."
    assert (
        len(bias_output[0].shape) == 2
    ), "Bias model profile head is of incorrect shape (None,out_pred_len) expected."

    # Combine outputs
    profile_out = Add(name=f"{name_prefix}_logits_profile_predictions")(
        [output_wo_bias[0], bias_output[0]]
    )
    concat_counts = Concatenate(axis=-1)([output_wo_bias[1], bias_output[1]])
    count_out = Lambda(
        lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_logcount_predictions",
    )(concat_counts)

    # Instantiate and compile the model
    full_model = Model(
        inputs=[inp], outputs=[profile_out, count_out], name=f"{name_prefix}_full_model"
    )
    full_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=[multinomial_nll, "mse"],
        loss_weights=[1, counts_loss_weight],
    )

    return full_model


def save_model_without_bias(model, output_prefix):
    """
    Saves the model without the bias component to a specified path.
    """
    # Assuming 'model_wo_bias' is a correctly named layer within your 'model'
    model_wo_bias_output = model.get_layer("chrombpnet_model").output
    model_without_bias = Model(inputs=model.input, outputs=model_wo_bias_output)
    print("Saving model without bias")
    model_without_bias.save(f"{output_prefix}nobias.h5")


#########################################
# evaluation
#########################################


def write_predictions_h5py(output_prefix, profile, logcts, coords):
    # open h5 file for writing predictions
    output_h5_fname = "{}_predictions.h5".format(output_prefix)
    h5_file = h5py.File(output_h5_fname, "w")
    # create groups
    coord_group = h5_file.create_group("coords")
    pred_group = h5_file.create_group("predictions")

    num_examples = len(coords)

    coords_chrom_dset = [str(coords[i][0]) for i in range(num_examples)]
    coords_center_dset = [int(coords[i][1]) for i in range(num_examples)]
    coords_peak_dset = [int(coords[i][3]) for i in range(num_examples)]

    dt = h5py.special_dtype(vlen=str)

    # create the "coords" group datasets
    coords_chrom_dset = coord_group.create_dataset(
        "coords_chrom",
        data=np.array(coords_chrom_dset, dtype=dt),
        dtype=dt,
        compression="gzip",
    )
    coords_start_dset = coord_group.create_dataset(
        "coords_center", data=coords_center_dset, dtype=int, compression="gzip"
    )
    coords_end_dset = coord_group.create_dataset(
        "coords_peak", data=coords_peak_dset, dtype=int, compression="gzip"
    )

    # create the "predictions" group datasets
    profs_dset = pred_group.create_dataset(
        "profs", data=profile, dtype=float, compression="gzip"
    )
    logcounts_dset = pred_group.create_dataset(
        "logcounts", data=logcts, dtype=float, compression="gzip"
    )

    # close hdf5 file
    h5_file.close()


def predict_on_batch_wrapper(model, test_generator):
    num_batches = len(test_generator)
    profile_probs_predictions = []
    true_counts = []
    counts_sum_predictions = []
    true_counts_sum = []
    coordinates = []

    for idx in tqdm(range(num_batches)):
        if idx % 100 == 0:
            print(str(idx) + "/" + str(num_batches))

        X, y, coords = test_generator[idx]

        # get the model predictions
        preds = model.predict_on_batch(X)

        # get counts predictions
        true_counts.extend(y[0])
        profile_probs_predictions.extend(softmax(preds[0]))

        # get profile predictions
        true_counts_sum.extend(y[1][:, 0])
        counts_sum_predictions.extend(preds[1][:, 0])
        coordinates.extend(coords)

    return (
        np.array(true_counts),
        np.array(profile_probs_predictions),
        np.array(true_counts_sum),
        np.array(counts_sum_predictions),
        np.array(coordinates),
    )


def _fix_sum_to_one(probs):
    """
    Fix probability arrays whose sum is fractinally above or
    below 1.0

    Args:
        probs (numpy.ndarray): An array whose sum is almost equal
            to 1.0

    Returns:
        np.ndarray: array that sums to 1
    """

    _probs = np.copy(probs)

    if np.sum(_probs) > 1.0:
        _probs[np.argmax(_probs)] -= np.sum(_probs) - 1.0

    if np.sum(_probs) < 1.0:
        _probs[np.argmin(_probs)] += 1.0 - np.sum(_probs)

    return _probs


def density_scatter(x, y, xlab, ylab, ax=None, sort=True, bins=20):
    """
    Scatter plot colored by 2d histogram
    """
    bad_indices = np.where(np.isnan(x)) + np.where(np.isnan(y))
    x = x[~np.isin(np.arange(x.size), bad_indices)]
    y = y[~np.isin(np.arange(y.size), bad_indices)]

    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z)

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel("Density")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return ax


# https://github.com/kundajelab/basepairmodels/blob/cf8e346e9df1bad9e55bd459041976b41207e6e5/basepairmodels/cli/metrics.py#L18
# replacing TracebackExceptions with assertions
def mnll(true_counts, logits=None, probs=None):
    """
    Compute the multinomial negative log-likelihood between true
    counts and predicted values of a BPNet-like profile model

    One of `logits` or `probs` must be given. If both are
    given `logits` takes preference.
    Args:
        true_counts (numpy.array): observed counts values

        logits (numpy.array): predicted logits values

        probs (numpy.array): predicted values as probabilities

    Returns:
        float: cross entropy

    """

    dist = None

    if logits is not None:
        # check for length mismatch
        assert len(logits) == len(true_counts)

        # convert logits to softmax probabilities
        probs = logits - logsumexp(logits)
        probs = np.exp(probs)

    elif probs is not None:
        # check for length mistmatch
        assert len(probs) == len(true_counts)

        # check if probs sums to 1
        # why is this nans sometimes
        assert abs(1.0 - np.sum(probs)) < 1e-1

    else:
        # both 'probs' and 'logits' are None
        print("At least one of probs or logits must be provided. " "Both are None.")

    # compute the nmultinomial distribution
    mnom = multinomial(np.sum(true_counts), probs)
    return -(mnom.logpmf(true_counts) / len(true_counts))


# https://github.com/kundajelab/basepairmodels/blob/cf8e346e9df1bad9e55bd459041976b41207e6e5/basepairmodels/cli/metrics.py#L129
def get_min_max_normalized_value(val, minimum, maximum):
    ret_val = (val - maximum) / (minimum - maximum)

    if ret_val < 0:
        return 0

    if ret_val > 1:
        return 1
    return ret_val


# https://github.com/kundajelab/basepairmodels/blob/cf8e346e9df1bad9e55bd459041976b41207e6e5/basepairmodels/cli/fastpredict.py#L59
def mnll_min_max_bounds(profile):
    """
    Min Max bounds for the mnll metric

    Args:
        profile (numpy.ndarray): true profile
    Returns:
        tuple: (min, max) bounds values
    """

    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile = profile.astype(np.float64)

    # profile as probabilities
    profile_prob = profile / np.sum(profile)

    # the scipy.stats.multinomial function is very sensitive to
    # profile_prob summing to exactly 1.0, if not you get NaN as the
    # resuls. In majority of the cases we can fix that problem by
    # adding or substracting the difference (but unfortunately it
    # doesnt always and there are cases where we still see NaNs, and
    # those we'll set to 0)
    profile_prob = _fix_sum_to_one(profile_prob)
    # print(profile, profile_prob)

    # mnll of profile with itself
    min_mnll = mnll(profile, probs=profile_prob)

    # if we still find a NaN, even after the above fix, set it to zero
    if math.isnan(min_mnll):
        min_mnll = 0.0

    if math.isinf(min_mnll):
        min_mnll = 0.0

    # mnll of profile with uniform profile
    max_mnll = mnll(profile, probs=uniform_profile)

    return (min_mnll, max_mnll)


# https://github.com/kundajelab/basepairmodels/blob/cf8e346e9df1bad9e55bd459041976b41207e6e5/basepairmodels/cli/fastpredict.py#L131
def jsd_min_max_bounds(profile):
    """
    Min Max bounds for the jsd metric

    Args:
        profile (numpy.ndarray): true profile

    Returns:
        tuple: (min, max) bounds values
    """

    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile_prob = profile / np.sum(profile)

    # jsd of profile with uniform profile
    max_jsd = jensenshannon(profile_prob, uniform_profile)

    # jsd of profile with itself (upper bound)
    min_jsd = 0.0

    return (min_jsd, max_jsd)


def counts_metrics(labels, preds, outf, title):
    """
    Get count metrics
    """
    spearman_cor = spearmanr(labels, preds)[0]
    pearson_cor = pearsonr(labels, preds)[0]
    mse = ((labels - preds) ** 2).mean(axis=0)

    return spearman_cor, pearson_cor, mse


def profile_metrics(true_counts, pred_probs, pseudocount=0.001):
    """
    Get profile metrics
    """
    mnll_pw = []
    mnll_norm = []

    jsd_pw = []
    jsd_norm = []
    jsd_rnd = []
    jsd_rnd_norm = []
    mnll_rnd = []
    mnll_rnd_norm = []

    num_regions = true_counts.shape[0]
    for idx in tqdm(range(num_regions)):
        # jsd
        cur_jsd = jensenshannon(
            true_counts[idx, :] / (pseudocount + np.nansum(true_counts[idx, :])),
            pred_probs[idx, :],
        )
        jsd_pw.append(cur_jsd)
        # normalized jsd
        min_jsd, max_jsd = jsd_min_max_bounds(true_counts[idx, :])
        curr_jsd_norm = get_min_max_normalized_value(cur_jsd, min_jsd, max_jsd)
        jsd_norm.append(curr_jsd_norm)

        # get random shuffling on labels for a worst case performance on metrics - labels versus shuffled labels
        shuffled_labels = np.random.permutation(true_counts[idx, :])
        shuffled_labels_prob = shuffled_labels / (
            pseudocount + np.nansum(shuffled_labels)
        )

        # mnll random
        # curr_rnd_mnll = mnll(true_counts[idx,:],  probs=shuffled_labels_prob)
        # mnll_rnd.append(curr_rnd_mnll)
        # normalized mnll random
        # curr_rnd_mnll_norm = get_min_max_normalized_value(curr_rnd_mnll, min_mnll, max_mnll)
        # mnll_rnd_norm.append(curr_rnd_mnll_norm)

        # jsd random
        curr_jsd_rnd = jensenshannon(
            true_counts[idx, :] / (pseudocount + np.nansum(true_counts[idx, :])),
            shuffled_labels_prob,
        )
        jsd_rnd.append(curr_jsd_rnd)
        # normalized jsd random
        curr_rnd_jsd_norm = get_min_max_normalized_value(curr_jsd_rnd, min_jsd, max_jsd)
        jsd_rnd_norm.append(curr_rnd_jsd_norm)

    return (
        np.array(mnll_pw),
        np.array(mnll_norm),
        np.array(jsd_pw),
        np.array(jsd_norm),
        np.array(jsd_rnd),
        np.array(jsd_rnd_norm),
        np.array(mnll_rnd),
        np.array(mnll_rnd_norm),
    )


def softmax(x, temp=1):
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp * norm_x) / np.sum(np.exp(temp * norm_x), axis=1, keepdims=True)
