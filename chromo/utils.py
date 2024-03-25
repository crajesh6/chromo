import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import sh
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from keras import backend as K
from sklearn.metrics import mean_squared_error
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from tfomics import moana, impress
from chromo import chrombpnet_utils
# from chrombpnet_utils import multinomial_nll


def make_directory(dir_name: str):
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    return


def Spearman(y_true, y_pred):
    return tf.py_function(
        spearmanr,
        [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
        Tout=tf.float32,
    )


def pearson_r(y_true, y_pred):
    # use smoothing for not resulting in NaN values
    # pearson correlation coefficient
    # https://github.com/WenYanger/Keras_Metrics
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)


# create functions
def summary_statistics(model, X, Y):
    pred = model.predict(X, batch_size=64)

    mse = mean_squared_error(Y[:, 0], pred[:, 0])
    pcc = stats.pearsonr(Y[:, 0], pred[:, 0])[0]
    scc = stats.spearmanr(Y[:, 0], pred[:, 0])[0]

    print("MSE = " + str("{0:0.4f}".format(mse)))
    print("PCC = " + str("{0:0.4f}".format(pcc)))
    print("SCC = " + str("{0:0.4f}".format(scc)))

    return mse, pcc, scc


def evaluate_model(model, X, Y, task):
    i = 0 if task == "Dev" else 1

    pred = model.predict(X, batch_size=64)  # [i].squeeze()
    if len(pred) == 2:
        pred = pred[i].squeeze()
    else:
        pred = pred[:, i]

    mse = mean_squared_error(Y[:, i], pred)
    pcc = stats.pearsonr(Y[:, i], pred)[0]
    scc = stats.spearmanr(Y[:, i], pred)[0]

    print(f"{task} MSE = {str('{0:0.3f}'.format(mse))}")
    print(f"{task} PCC = {str('{0:0.3f}'.format(pcc))}")
    print(f"{task} SCC = {str('{0:0.3f}'.format(scc))}")

    return mse, pcc, scc


def evaluate_model_deepstarr(model, test_data, evaluation_path):
    x_test, y_test = test_data
    mse_dev, pcc_dev, scc_dev = evaluate_model(model, x_test, y_test, "Dev")
    mse_hk, pcc_hk, scc_hk = evaluate_model(model, x_test, y_test, "Hk")

    data = [
        {
            "MSE_dev": mse_dev,
            "PCC_dev": pcc_dev,
            "SCC_dev": scc_dev,
            "MSE_hk": mse_hk,
            "PCC_hk": pcc_hk,
            "SCC_hk": scc_hk,
        }
    ]
    df = pd.DataFrame(data)
    pd.DataFrame(df).to_csv(f"{evaluation_path}/model_performance.csv", index=False)
    return df


def evaluate_model_hepg2(model, test_data, evaluation_path):
    x_test, y_test = test_data
    mse, pcc, scc = evaluate_model(model, x_test, y_test, "Dev")

    data = [
        {
            "MSE": mse,
            "PCC": pcc,
            "SCC": scc,
        }
    ]
    df = pd.DataFrame(data)
    pd.DataFrame(df).to_csv(f"{evaluation_path}/model_performance.csv", index=False)
    return df


def evaluate_model_chrombpnet(model, test_gen, evaluation_path=None):
    metrics_dictionary = {"counts_metrics": {}, "profile_metrics": {}}

    (
        true_counts,
        profile_probs_predictions,
        true_counts_sum,
        counts_sum_predictions,
        coordinates,
    ) = chrombpnet_utils.predict_on_batch_wrapper(model, test_gen)

    print("Finished saving predictions!")

    (
        mnll_pw,
        mnll_norm,
        jsd_pw,
        jsd_norm,
        jsd_rnd,
        jsd_rnd_norm,
        mnll_rnd,
        mnll_rnd_norm,
    ) = chrombpnet_utils.profile_metrics(true_counts, profile_probs_predictions)
    peaks_idx = coordinates[:, 3] == "1"
    spearman_cor, pearson_cor, mse = chrombpnet_utils.counts_metrics(
        true_counts_sum[peaks_idx], counts_sum_predictions[peaks_idx], None, None
    )

    metrics_dictionary["counts_metrics"]["peaks"] = {}
    metrics_dictionary["counts_metrics"]["peaks"]["spearmanr"] = spearman_cor
    metrics_dictionary["counts_metrics"]["peaks"]["pearsonr"] = pearson_cor
    metrics_dictionary["counts_metrics"]["peaks"]["mse"] = mse

    metrics_dictionary["profile_metrics"]["peaks"] = {}
    metrics_dictionary["profile_metrics"]["peaks"]["median_jsd"] = np.nanmedian(
        jsd_pw[peaks_idx]
    )
    metrics_dictionary["profile_metrics"]["peaks"]["median_norm_jsd"] = np.nanmedian(
        jsd_norm[peaks_idx]
    )

    # save metrics_dictionary
    counts_df = pd.DataFrame(metrics_dictionary["counts_metrics"])
    profile_df = pd.DataFrame(metrics_dictionary["profile_metrics"])
    
    if evaluation_path:
        counts_df.to_csv(f"{evaluation_path}/counts_metrics.csv")
        profile_df.to_csv(f"{evaluation_path}/profile_metrics.csv")

    # Need to concat these into 1 eval df

    return counts_df, profile_df


def get_cor(data):
    samples = ["all", "At", "Zm", "Sb"]

    rsquare = []
    spearman = []

    for species in samples:
        if species == "all":
            data_filt = data
        else:
            data_filt = data[data["sample.name"] == species]

        rsquare.append(
            round(data_filt["enrichment"].corr(data_filt["prediction"]) ** 2, 2)
        )
        spearman.append(
            round(
                data_filt["enrichment"].corr(
                    data_filt["prediction"], method="spearman"
                ),
                2,
            )
        )

    return pd.DataFrame(
        {"sample.name": samples, "spearman": spearman, "rsquare": rsquare}
    )


def evaluate_model_plantstarr(model, test_data, evaluation_path, dataset="leaf"):
    x_test, y_test = test_data

    test_df_path = (
        f"/home/chandana/projects/deepstarr/data/plantstarr/CNN_test_{dataset}.tsv"
    )
    df = pd.read_csv(test_df_path, sep="\t", header=0)

    pred = model.predict(x_test, batch_size=64)
    df["prediction"] = pred
    df = df.rename(columns={"sp": "sample.name"}).sample(frac=1)

    df = get_cor(df)

    print(f"Training data in plant {dataset} system:")
    print(df)

    pd.DataFrame(df).to_csv(f"{evaluation_path}/model_performance.csv", index=False)
    # df = pd.DataFrame()
    return df


def _evaluate_model_scbasset(model, test_data, evaluation_path):
    x_test, y_test = test_data

    loss, auroc, aupr = model.evaluate(x_test, y_test)

    data = [
        {
            "loss": loss,
            "auroc": auroc,
            "aupr": aupr,
        }
    ]
    df = pd.DataFrame(data)
    pd.DataFrame(df).to_csv(f"{evaluation_path}/model_performance.csv", index=False)
    return df


def evaluate_model_scbasset(model, test_data, evaluation_path):
    x_test, y_test = test_data

    num_peaks = x_test.shape[0]
    num_cells = model.layers[-1].output_shape[-1]
    y_pred = model.predict(x_test, batch_size=64, verbose=0)

    # per cell metrics:
    cell_results = []

    for i in tqdm(range(num_cells)):
        auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")
        aupr = tf.keras.metrics.AUC(curve="PR", name="aupr")

        cell_results += [
            {
                "cell_type": i,
                "auroc": auroc(y_test[:, i], y_pred[:, i]).numpy(),
                "aupr": aupr(y_test[:, i], y_pred[:, i]).numpy(),
            }
        ]
    cell_results = pd.DataFrame(cell_results)

    # per peak metrics:
    peak_results = []

    for i in tqdm(range(num_peaks)):
        auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")
        aupr = tf.keras.metrics.AUC(curve="PR", name="aupr")

        peak_results += [
            {
                "peak_index": i,
                "auroc": auroc(y_test[i, :], y_pred[i, :]).numpy(),
                "aupr": aupr(y_test[i, :], y_pred[i, :]).numpy(),
            }
        ]

    peak_results = pd.DataFrame(peak_results)

    loss, auroc, aupr = model.evaluate(x_test, y_test)

    data = [
        {
            "loss": loss,
            "auroc": auroc,
            "aupr": aupr,
            "per_cell_auroc": cell_results["auroc"].mean(),
            "per_cell_aupr": cell_results["aupr"].mean(),
            "per_peak_auroc": peak_results["auroc"].mean(),
            "per_peak_aupr": peak_results["aupr"].mean(),
        }
    ]
    df = pd.DataFrame(data)
    pd.DataFrame(df).to_csv(f"{evaluation_path}/model_performance.csv", index=False)
    return df


def calculate_filter_activations(
    model,
    x_test,
    params_path,
    layer,
    threshold,
    window,
    batch_size=64,
    plot_filters=False,
    from_saved=False,
):
    """Calculate filter activations and return the final results."""
    filters_path = os.path.join(params_path, "filters")

    if not from_saved:
        make_directory(filters_path)

        print("Making intermediate predictions...")

        # Get the intermediate layer model
        intermediate = tf.keras.Model(
            inputs=model.inputs, outputs=model.layers[layer].output
        )

        # Open the file to append predictions
        predictions_file = f"{filters_path}/filters_{layer}.pkl"

        # Remove the file if it already exists:
        path = Path(predictions_file)
        if path.is_file():
            sh.rm(predictions_file)

        with open(predictions_file, "ab") as file:
            # Iterate over batches
            for batch in batch_generator(x_test, batch_size):
                # Get predictions for the batch
                batch_predictions = intermediate.predict(batch, verbose=0)

                # Append predictions to the pickle file
                pickle.dump(batch_predictions, file)

        # Load predictions from the pickle file
        fmap = load_predictions_from_file(predictions_file)

        # Remove predictions file:
        path = Path(predictions_file)
        if path.is_file():
            print("Now removing the predictions file!")
            sh.rm(predictions_file)

        # Concatenate the predictions into a single array
        fmap = np.concatenate(fmap, axis=0)

        # Perform further calculations on fmap
        print("Calculating filter activations...")
        W, counts = activation_pwm(fmap, x_test, threshold=threshold, window=window)

        # Filter out empty filters:
        # Check if batches have all elements not equal to 0.25
        batch_flags = np.all(W != 0.25, axis=2)

        # Find the indices of batches with all elements not equal to 0.25
        batch_indices = np.where(batch_flags.sum(axis=-1))[0]
        print(f"Learned filters : empty filters = {len(batch_indices)} : {len(W)}")

        # Select batches from the original array
        sub_W = W[batch_indices]

        # Clip filters for TomTom
        W_clipped = clip_filters(sub_W, threshold=0.5, pad=3)
        moana.meme_generate(
            W_clipped, output_file=f"{filters_path}/filters_{layer}.txt"
        )

        # save filter PWMs to an h5 file
        with h5py.File(f"{filters_path}/filters_{layer}.h5", "w") as f:
            dset = f.create_dataset(name="filters", data=W, dtype="float32")
            dset = f.create_dataset(name="filters_subset", data=sub_W, dtype="float32")
            dset = f.create_dataset(name="counts", data=counts, dtype="float32")

        # write jaspar file for RSAT:
        print("Writing output for RSAT...")
        output_file = f"{filters_path}/filters_{layer}_hits.jaspar"

        path = Path(output_file)
        if path.is_file():
            sh.rm(output_file)

        # get the position frequency matrix
        pfm = np.array([W[i] * counts[i] for i in range(len(counts))])

        # write jaspar file for RSAT:
        write_filters_jaspar(output_file, pfm, batch_indices)

        # run RSAT:
        run_rsat(params_path)

    # Load filter PWMs, counts from an h5 file
    print("Loading filters...")
    with h5py.File(f"{filters_path}/filters_{layer}.h5", "r") as f:
        W = f["filters"][:]
        sub_W = f["filters_subset"][:]
        counts = f["filters"][:]

    # Plot filters
    if plot_filters:
        print("Plotting filters...")
        filters_fig_path = os.path.join(filters_path, f"filters_{layer}.pdf")
        plot_filters_and_return_path(W, filters_fig_path, indices=range(len(W)))

        print("Plotting non-redundant filters...")
        # get non-redundant set of motifs:
        clustered_filters_path = (
            f"{filters_path}/clustered_filters/clustered_filters_central_motifs_IDs.tab"
        )
        non_redundant_filters_fig_path = os.path.join(
            filters_path, f"non_redundant_filters_{layer}.pdf"
        )

        non_redundant_filters = pd.read_csv(
            clustered_filters_path, sep="\t", header=None
        )[2].to_list()
        non_redundant_filters = [
            int(i.split("filter_")[-1]) for i in non_redundant_filters
        ]
        plot_filters_and_return_path(
            W[non_redundant_filters],
            non_redundant_filters_fig_path,
            indices=non_redundant_filters,
        )

    return W, sub_W, counts


def plot_filters_and_return_path(W, filters_fig_path, indices, threshold=False):
    """Plot filters and return the path to the saved figure."""
    num_plot = len(W)
    # Plot filters
    fig = plt.figure(figsize=(20, num_plot // 10))
    impress.plot_filters(W, fig, num_cols=10, fontsize=12, names=indices)

    fig.savefig(filters_fig_path, format="pdf", dpi=200, bbox_inches="tight")

    return


def batch_generator(data, batch_size):
    """Generate batches from data."""
    num_batches = len(data) // batch_size
    for i in tqdm(range(num_batches)):
        yield data[i * batch_size : (i + 1) * batch_size]


def load_predictions_from_file(file_path):
    """Load predictions from a pickle file."""
    predictions = []
    with open(file_path, "rb") as file:
        while True:
            try:
                batch_predictions = pickle.load(file)
                predictions.append(batch_predictions)
            except EOFError:
                break
    return predictions


def activation_pwm(fmap, X, threshold=0.5, window=20):
    # extract sequences with aligned activation
    window_left = int(window / 2)
    window_right = window - window_left

    N, L, A = X.shape
    num_filters = fmap.shape[-1]

    W = []
    counts = []
    for filter_index in tqdm(range(num_filters)):
        counter = 0

        # find regions above threshold
        coords = np.where(
            fmap[:, :, filter_index] > np.max(fmap[:, :, filter_index]) * threshold
        )

        if len(coords) > 1:
            x, y = coords

            # sort score
            index = np.argsort(fmap[x, y, filter_index])[::-1]
            data_index = x[index].astype(int)
            pos_index = y[index].astype(int)

            # make a sequence alignment centered about each activation (above threshold)
            seq_align = []
            for i in range(len(pos_index)):
                # determine position of window about each filter activation
                start_window = pos_index[i] - window_left
                end_window = pos_index[i] + window_right

                # check to make sure positions are valid
                if (start_window > 0) & (end_window < L):
                    seq = X[data_index[i], start_window:end_window, :]
                    seq_align.append(seq)
                    counter += 1

            # calculate position probability matrix
            if len(seq_align) > 1:  # try:
                W.append(np.mean(seq_align, axis=0))
            else:
                W.append(np.ones((window, 4)) / 4)
        else:
            W.append(np.ones((window, 4)) / 4)
        counts.append(counter)
    return np.array(W), np.array(counts)


@tf.function
def saliency_map(X, model, class_index=0):
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        outputs = model(X)[:, class_index]
    grad = tape.gradient(outputs, X)
    return grad


def write_filters_jaspar(output_file, pfm, batch_indices):
    # open file for writing
    f = open(output_file, "w")
    sub_pfm = pfm[batch_indices]
    for i, pwm in enumerate(sub_pfm):
        f.write(f">filter_{batch_indices[i]} filter_{batch_indices[i]}\n")

        for j, base in enumerate("ACGT"):
            terms = [f"{value:6.2f}" for value in pwm.T[j]]
            line = f"{base} [{' '.join(terms)}]\n"

            f.write(line)

    f.close()

    return


def clip_filters(W, threshold=0.5, pad=3):
    """clip uninformative parts of conv filters"""
    W_clipped = []
    for w in W:
        L, A = w.shape
        entropy = np.log2(4) + np.sum(w * np.log2(w + 1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index) - pad, 0)
            end = np.minimum(np.max(index) + pad + 1, L)
            W_clipped.append(w[start:end, :])
        else:
            W_clipped.append(w)

    return W_clipped


def meme_generate(W, output_file="meme.txt", prefix="filter"):
    """generate a meme file for a set of filters, W ∈ (N,L,A)"""

    # background frequency
    nt_freqs = [1.0 / 4 for i in range(4)]

    # open file for writing
    f = open(output_file, "w")

    # print intro material
    f.write("MEME version 4\n")
    f.write("\n")
    f.write("ALPHABET= ACGT\n")
    f.write("\n")
    f.write("Background letter frequencies:\n")
    f.write("A %.4f C %.4f G %.4f T %.4f \n" % tuple(nt_freqs))
    f.write("\n")

    for j, pwm in enumerate(W):
        L, A = pwm.shape
        f.write("MOTIF %s%d \n" % (prefix, j))
        f.write("letter-probability matrix: alength= 4 w= %d nsites= %d \n" % (L, L))
        for i in range(L):
            f.write("%.4f %.4f %.4f %.4f \n" % tuple(pwm[i, :]))
        f.write("\n")

    f.close()


def run_rsat(save_path):
    # Run the command using conda run
    sh.conda.run(
        "-n", "rsat", "/home/chandana/projects/homo/scripts/run_rsat.sh", save_path
    )

    return


class AttrDict(dict):
    """A dictionary that allows attribute-like access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value


def load_config(config_path):
    config = yaml.full_load(open(config_path))
    return AttrDict(config)


def get_model_param_dict(args):
    params = {}
    for line in open(args["params"], "r").read().strip().split("\n"):
        tokens = line.split("\t")
        params[tokens[0]] = tokens[1]

    return params


def dna_to_one_hot(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
    position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
    of N strings, where every string is the same length L. Returns an N x L x 4
    NumPy array of one-hot encodings, in the same order as the input sequences.
    All bases will be converted to upper-case prior to performing the encoding.
    Any bases that are not "ACGT" will be given an encoding of all 0s.
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    # Join all sequences together into one long string, all uppercase
    seq_concat = "".join(seqs).upper() + "ACGT"
    # Add one example of each base, so np.unique doesn't miss indices later

    one_hot_map = np.identity(5)[:, :-1].astype(np.int8)

    # Convert string into array of ASCII character codes;
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

    # Anything that's not an A, C, G, or T gets assigned a higher code
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85

    # Convert the codes into indices in [0, 4], in ascending order by code
    _, base_inds = np.unique(base_vals, return_inverse=True)

    # Get the one-hot encoding for those indices, and reshape back to separate
    return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))


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


def filter_edge_regions(peaks_df, bw, width, peaks_bool):
    """
    Filter regions in bed file that are on edges i.e regions that cannot be used to construct
    input length + jitter length of the sequence
    """
    input_shape = peaks_df.shape[0]

    # left edge case
    filtered = np.array((peaks_df["start"] + peaks_df["summit"] - width // 2) < 0)
    peaks_df = peaks_df[~filtered]
    num_filtered = sum(filtered)

    # right edge case
    chrom_to_sizes = bw.chroms()
    filtered = []
    for i, r in peaks_df.iterrows():
        if r["start"] + r["summit"] + width // 2 > chrom_to_sizes.get(r["chr"], 0):
            filtered.append(True)
        else:
            filtered.append(False)
    filtered = np.array(filtered)
    peaks_df = peaks_df[~filtered]
    num_filtered += sum(filtered)

    if peaks_bool:
        print("Number of peaks input: ", input_shape)
        print(
            "Number of peaks filtered because the input/output is on the edge: ",
            num_filtered,
        )
        print("Number of peaks being used: ", peaks_df.shape[0])
    else:
        print("Number of non peaks input: ", input_shape)
        print(
            "Number of non peaks filtered because the input/output is on the edge: ",
            num_filtered,
        )
        print("Number of non peaks being used: ", peaks_df.shape[0])
    return peaks_df


def get_seqs_cts(genome, bw, peaks_df, input_width=2114, output_width=1000):
    """
    Output counts (not log counts)
    Output one-hot encoded sequence
    """
    vals = []
    seqs = []
    for i, r in peaks_df.iterrows():
        sequence = str(
            genome[r["chr"]][
                (r["start"] + r["summit"] - input_width // 2) : (
                    r["start"] + r["summit"] + input_width // 2
                )
            ]
        )
        seqs.append(sequence)
        bigwig_vals = np.nan_to_num(
            bw.values(
                r["chr"],
                (r["start"] + r["summit"]) - output_width // 2,
                (r["start"] + r["summit"]) + output_width // 2,
            )
        )
        vals.append(bigwig_vals)
    return (np.sum(np.array(vals), axis=1), dna_to_one_hot(seqs))


def load_model_wrapper(model_h5):
    # read .h5 model
    custom_objects = {"tf": tf, "multinomial_nll": chrombpnet_utils.multinomial_nll}
    get_custom_objects().update(custom_objects)
    model = load_model(model_h5)
    print("got the model")
    model.summary()
    return model
