import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt


def absmaxND(a, axis=None):
    amax = np.max(a, axis)
    amin = np.min(a, axis)
    return np.where(-amin > amax, amin, amax)


def get_layer_output(model, index, X):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    return temp.predict(X)


def pearsonr(vector1, vector2):
    m1 = np.mean(vector1)
    m2 = np.mean(vector2)

    diff1 = vector1 - m1
    diff2 = vector2 - m2

    top = np.sum(diff1 * diff2)
    bottom = np.sum(np.power(diff1, 2)) * np.sum(np.power(diff2, 2))
    bottom = np.sqrt(bottom)

    return top / bottom


def plot_glifac(ax, correlation_matrix, filter_labels, vmin=-0.5, vmax=0.5):
    ax.set_xticks(list(range(len(filter_labels))))
    ax.set_yticks(list(range(len(filter_labels))))
    ax.set_xticklabels(filter_labels, rotation=90, size=10)
    ax.set_yticklabels(filter_labels, size=10)
    c = ax.imshow(correlation_matrix, cmap="bwr_r", vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(c, ax=ax, shrink=0.5)
    cbar.set_label(
        "Correlation", labelpad=5, ha="center", va="center", rotation=-90, size=15
    )

    # Remove the x-axis and y-axis lines
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax, cbar


def correlation_matrix(
    model,
    c_index,
    mha_index,
    X,
    thresh=0.1,
    random_frac=0.5,
    limit=None,
    head_concat=np.max,
    symmetrize=absmaxND,
):
    """
    * model                  trained tensorflow model
    * c_index                index of the convolutoinal layer (after pooling)
    * mha_index              index of multi-head attention layer
    * X                      test sequences
    * thresh                 attention threshold
    * random_frac            proportion of negative positions in the set of position interactions
    * limit                  maximum number of position interactions processed; sometimes needed to avoid resource exhaustion
    * head_concat            function for concatenating heads; e.g. np.max, np.mean
    * symmetrize             function for symmetrizing the correlation matrix across diagonal
    """

    assert 0 <= random_frac < 1

    feature_maps = get_layer_output(model, c_index, X)
    o, att_maps = get_layer_output(model, mha_index, X)
    att_maps = head_concat(att_maps, axis=1)

    position_interactions = get_position_interactions(att_maps, thresh)
    num_rands = int(random_frac / (1 - random_frac))
    random_interactions = [
        np.random.randint(len(att_maps), size=(num_rands, 1)),
        np.random.randint(att_maps.shape[1], size=(num_rands, 2)),
    ]
    position_pairs = [
        np.vstack([position_interactions[0], random_interactions[0]]),
        np.vstack([position_interactions[1], random_interactions[1]]),
    ]
    if limit is not None:
        permutation = np.random.permutation(len(position_pairs[0]))
        position_pairs = [
            position_pairs[0][permutation],
            position_pairs[1][permutation],
        ]
        position_pairs = [position_pairs[0][:limit], position_pairs[1][:limit]]

    filter_interactions = feature_maps[position_pairs].transpose([1, 2, 0])
    correlation_matrix = correlation(filter_interactions[0], filter_interactions[1])
    if symmetrize is not None:
        correlation_matrix = symmetrize(
            np.array([correlation_matrix, correlation_matrix.transpose()]), axis=0
        )
    correlation_matrix = np.nan_to_num(correlation_matrix)

    return correlation_matrix


def get_position_interactions(att_maps, threshold=0.1):
    position_interactions = np.array(np.where(att_maps >= threshold))
    position_interactions = [
        position_interactions[[0]].transpose(),
        position_interactions[[1, 2]].transpose(),
    ]
    return position_interactions


def correlation(set1, set2, function=pearsonr):
    combinations = (
        np.indices(dimensions=(set1.shape[0], set2.shape[0]))
        .transpose()
        .reshape((-1, 2))
        .transpose()[::-1]
    )
    vector_mesh = [set1[combinations[0]], set2[combinations[1]]]
    vector_mesh = np.array(vector_mesh).transpose([1, 0, 2])
    correlations = []
    for i in tqdm(range(len(vector_mesh))):
        r = function(vector_mesh[i][0], vector_mesh[i][1])
        correlations.append(r)
    correlations = np.array(correlations).reshape((len(set1), len(set2)))
    return correlations
