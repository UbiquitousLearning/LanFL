"""Helper to visualize metrics-2-1."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
import pandas as pd
import sys

from decimal import Decimal

models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(models_dir)

ACCURACY_KEY = 'accuracy'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
SERVER_TIME_KEY = 'server_time'


# def load_data(stat_metrics_file='stat_metrics.csv', sys_metrics_file='sys_metrics.csv'):
def load_data(stat_metrics_file='stat_metrics.csv'):
    """Loads the data from the given stat_metric and sys_metric files."""
    stat_metrics = pd.read_csv(stat_metrics_file) if stat_metrics_file else None

    if stat_metrics is not None:
        stat_metrics.sort_values(by=NUM_ROUND_KEY, inplace=True)

    return stat_metrics


def _set_plot_properties(properties):
    """Sets some plt properties."""
    if 'xlim' in properties:
        plt.xlim(properties['xlim'])
    if 'ylim' in properties:
        plt.ylim(properties['ylim'])
    if 'xlabel' in properties:
        plt.xlabel(properties['xlabel'])
    if 'ylabel' in properties:
        plt.ylabel(properties['ylabel'])


def get_acc(stat_metrics, round_or_time, weighted=False):
    if weighted:
        accuracies = stat_metrics.groupby(round_or_time).apply(_weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
        accuracies = accuracies.reset_index(name=ACCURACY_KEY)

    else:
        accuracies = stat_metrics.groupby(round_or_time, as_index=False).mean()

    return accuracies


def plot_accuracy_vs_round_number_leaf_lan_aware(weighted=False, plot_stds=False, figsize=(10, 8), title_fontsize=16, **kwargs):
    """Plots the clients' average test accuracy vs. the round number.
    Args:
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        title_fontsize: Font size for the plot's title.
        kwargs: Arguments to be passed to _set_plot_properties."""
    plt.figure(figsize=figsize)
    title_weighted = 'Weighted' if weighted else 'Unweighted'
    plt.title('Weighted Accuracy vs Round Number', fontsize=title_fontsize)

    ## 同构lan vs leaf， 固定lan的个数，leaf-2、lan-10-20-30
    ## 同构lan vs leaf， 固定lan的个数，leaf-2、lan-10-20-30
    # plt.plot(leaf_epochs_1_acc_round[NUM_ROUND_KEY], leaf_epochs_1_acc_round[ACCURACY_KEY],
    #          label='Leaf,E=1,WB=2',
    #          linestyle='-', color='c', linewidth=3)
    plt.plot(leaf_epochs_5_acc_round[NUM_ROUND_KEY], leaf_epochs_5_acc_round[ACCURACY_KEY],
             label='Leaf,E=5,WB=2',
             linestyle='-', color='m', linewidth=3)
    plt.plot(leaf_epochs_10_acc_round[NUM_ROUND_KEY], leaf_epochs_10_acc_round[ACCURACY_KEY],
             label='Leaf,E=10,WB=2,BS=5',
             linestyle='-', color='red', linewidth=3)
    # plt.plot(leaf_epochs_15_acc_round[NUM_ROUND_KEY], leaf_epochs_15_acc_round[ACCURACY_KEY],
    #          label='Leaf,E=15,WB=2',
    #          linestyle='-', color='b', linewidth=3)
    plt.plot(leaf_epochs_20_acc_round[NUM_ROUND_KEY], leaf_epochs_20_acc_round[ACCURACY_KEY],
             label='Leaf,E=20,WB=2',
             linestyle='-', color='c', linewidth=3)

    plt.plot(leaf_epochs_5_size_10_acc_round[NUM_ROUND_KEY], leaf_epochs_5_size_10_acc_round[ACCURACY_KEY],
             label='Leaf,E=5,WB=2,BS=10',
             linestyle='-.', color='m', linewidth=3)
    plt.plot(leaf_epochs_10_size_10_acc_round[NUM_ROUND_KEY], leaf_epochs_10_size_10_acc_round[ACCURACY_KEY],
             label='Leaf,E=10,WB=2,BS=10',
             linestyle='-.', color='red', linewidth=3)

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim(0, 500)

    plt.ylabel('Weighted Accuracy')
    plt.xlabel('Round Number')
    _set_plot_properties(kwargs)
    plt.show()


def plot_accuracy_vs_server_time_leaf_lan_aware(weighted=False, plot_stds=False, figsize=(10, 8), title_fontsize=16, **kwargs):
    """Plots the clients' average test accuracy vs. the round number.
    Args:
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        title_fontsize: Font size for the plot's title.
        kwargs: Arguments to be passed to _set_plot_properties."""
    plt.figure(figsize=figsize)
    title_weighted = 'Weighted' if weighted else 'Unweighted'
    plt.title('Weighted Accuracy vs Clock Time', fontsize=title_fontsize)
    # fig, ax = plt.subplots(1, 1, figsize=(6,4))

    ## 同构lan vs leaf， 固定lan的个数，leaf-2、lan-10-20-30
    # plt.plot([time_second / 3600 for time_second in leaf_epochs_1_acc_time[SERVER_TIME_KEY]],
    #          leaf_epochs_1_acc_time[ACCURACY_KEY],
    #          label='Leaf,E=1,WB=2',
    #          linestyle='-', color='c', linewidth=3)
    plt.plot([time_second / 3600 for time_second in leaf_epochs_5_acc_time[SERVER_TIME_KEY]],
             leaf_epochs_5_acc_time[ACCURACY_KEY],
             label='Leaf,E=5,WB=2',
             linestyle='-', color='m', linewidth=3)
    plt.plot([time_second/3600 for time_second in leaf_epochs_10_acc_time[SERVER_TIME_KEY]], leaf_epochs_10_acc_time[ACCURACY_KEY],
             label='Leaf,E=10,WB=2,BS=5',
             linestyle='-', color='red', linewidth=3)
    # plt.plot([time_second / 3600 for time_second in leaf_epochs_15_acc_time[SERVER_TIME_KEY]],
    #          leaf_epochs_15_acc_time[ACCURACY_KEY],
    #          label='Leaf,E=20,WB=2',
    #          linestyle='-', color='b', linewidth=3)
    plt.plot([time_second / 3600 for time_second in leaf_epochs_20_acc_time[SERVER_TIME_KEY]],
             leaf_epochs_20_acc_time[ACCURACY_KEY],
             label='Leaf,E=20,WB=2',
             linestyle='-', color='c', linewidth=3)

    plt.plot([time_second / 3600 for time_second in leaf_epochs_5_size_10_acc_time[SERVER_TIME_KEY]],
             leaf_epochs_5_size_10_acc_time[ACCURACY_KEY],
             label='Leaf,E=5,WB=2,BS=10',
             linestyle='-.', color='m', linewidth=3)
    plt.plot([time_second/3600 for time_second in leaf_epochs_10_size_10_acc_time[SERVER_TIME_KEY]], leaf_epochs_10_size_10_acc_time[ACCURACY_KEY],
             label='Leaf,E=10,WB=2,BS=10',
             linestyle='-.', color='red', linewidth=3)

    plt.legend(loc='lower right')
    # plt.xlim(0, 20)
    # plt.ylim(0.6, 0.9)
    plt.grid(True)

    plt.ylabel('Weighted Accuracy')
    plt.xlabel('Clock Time(Hour)')
    _set_plot_properties(kwargs)
    plt.show()


def get_x_y(list_to_simple):
    x = [list_to_simple[i] for i in range(0, len(list_to_simple), 10)]
    return x


def _weighted_mean(df, metric_name, weight_name):
    d = df[metric_name]
    w = df[weight_name]
    try:
        return (w * d).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


def _weighted_std(df, metric_name, weight_name):
    d = df[metric_name]
    w = df[weight_name]
    try:
        weigthed_mean = (w * d).sum() / w.sum()
        return (w * ((d - weigthed_mean) ** 2)).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


if __name__ == '__main__':
    # data dir

    ## 同构lan vs leaf， 固定lan的个数，leaf-2、lan-10-20-30
    # no batch_num
    # leaf_epochs_1_dir = 'lan-vs-leaf/leaf/clients_50/epochs_1/metrics/metrics_stat.csv'
    # leaf_epochs_5_dir = 'lan-vs-leaf/leaf/clients_50/epochs_5/metrics-1/metrics_stat.csv'
    # leaf_epochs_10_dir = 'lan-vs-leaf/leaf/clients_50/epochs_10/metrics-1/metrics_stat.csv'
    # # get accuracy and std
    # leaf_epochs_1_acc_round = get_acc(load_data(leaf_epochs_1_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_1_acc_time = get_acc(load_data(leaf_epochs_1_dir), SERVER_TIME_KEY, weighted=True)
    # leaf_epochs_5_acc_round = get_acc(load_data(leaf_epochs_5_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_5_acc_time = get_acc(load_data(leaf_epochs_5_dir), SERVER_TIME_KEY, weighted=True)
    # leaf_epochs_10_acc_round = get_acc(load_data(leaf_epochs_10_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_10_acc_time = get_acc(load_data(leaf_epochs_10_dir), SERVER_TIME_KEY, weighted=True)

    # leaf_epochs_1_dir = 'leaf_time/clients_10/batch_size_5/leaf_epochs_1/metrics/metrics_stat.csv'
    # leaf_epochs_2_dir = 'leaf_time/clients_10/batch_size_5/leaf_epochs_2/metrics/metrics_stat.csv'
    # leaf_epochs_5_dir = 'leaf_time/clients_10/batch_size_5/leaf_epochs_5/metrics/metrics_stat.csv'
    #
    # leaf_epochs_1_acc_round = get_acc(load_data(leaf_epochs_1_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_1_acc_time = get_acc(load_data(leaf_epochs_1_dir), SERVER_TIME_KEY, weighted=True)
    # leaf_epochs_2_acc_round = get_acc(load_data(leaf_epochs_2_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_2_acc_time = get_acc(load_data(leaf_epochs_2_dir), SERVER_TIME_KEY, weighted=True)
    # leaf_epochs_5_acc_round = get_acc(load_data(leaf_epochs_5_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_5_acc_time = get_acc(load_data(leaf_epochs_5_dir), SERVER_TIME_KEY, weighted=True)

    # leaf_epochs_1_dir = 'leaf_time/clients_10/batch_num_5/epochs_1/metrics/metrics_stat.csv'
    # leaf_epochs_5_dir = 'leaf_time/clients_10/batch_num_5/epochs_5/metrics/metrics_stat.csv'
    # leaf_epochs_20_dir = 'leaf_time/clients_10/batch_num_5/epochs_20/metrics/metrics_stat.csv'
    # # get accuracy and std
    # leaf_epochs_1_acc_round = get_acc(load_data(leaf_epochs_1_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_1_acc_time = get_acc(load_data(leaf_epochs_1_dir), SERVER_TIME_KEY, weighted=True)
    # leaf_epochs_5_acc_round = get_acc(load_data(leaf_epochs_5_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_5_acc_time = get_acc(load_data(leaf_epochs_5_dir), SERVER_TIME_KEY, weighted=True)
    # leaf_epochs_20_acc_round = get_acc(load_data(leaf_epochs_20_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_20_acc_time = get_acc(load_data(leaf_epochs_20_dir), SERVER_TIME_KEY, weighted=True)

    leaf_epochs_1_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_1/epochs_1/metrics/metrics_stat.csv'
    # batch_num=20 batch_size=5
    leaf_epochs_5_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_20/epochs_5/metrics/metrics_stat.csv'
    leaf_epochs_10_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_20/epochs_10/metrics/metrics_stat.csv'
    leaf_epochs_15_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_1/epochs_15/metrics/metrics_stat.csv'
    leaf_epochs_20_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_20/epochs_20/metrics/metrics_stat.csv'
    # batch_num=20 batch_size=10
    leaf_epochs_5_size_10_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_20/epochs_5/metrics-2/metrics_stat.csv'
    leaf_epochs_10_size_10_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_20/epochs_10/metrics-2/metrics_stat.csv'
    leaf_epochs_15_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_1/epochs_15/metrics/metrics_stat.csv'
    leaf_epochs_20_size_10_dir = 'lan-vs-leaf/leaf/clients_50/batch_num_20/epochs_20/metrics-2/metrics_stat.csv'
    # get accuracy and std
    # leaf_epochs_1_acc_round = get_acc(load_data(leaf_epochs_1_dir), NUM_ROUND_KEY, weighted=True)
    # leaf_epochs_1_acc_time = get_acc(load_data(leaf_epochs_1_dir), SERVER_TIME_KEY, weighted=True)
    leaf_epochs_5_acc_round = get_acc(load_data(leaf_epochs_5_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_5_acc_time = get_acc(load_data(leaf_epochs_5_dir), SERVER_TIME_KEY, weighted=True)
    leaf_epochs_10_acc_round = get_acc(load_data(leaf_epochs_10_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_10_acc_time = get_acc(load_data(leaf_epochs_10_dir), SERVER_TIME_KEY, weighted=True)
    leaf_epochs_15_acc_round = get_acc(load_data(leaf_epochs_15_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_15_acc_time = get_acc(load_data(leaf_epochs_15_dir), SERVER_TIME_KEY, weighted=True)
    leaf_epochs_20_acc_round = get_acc(load_data(leaf_epochs_20_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_20_acc_time = get_acc(load_data(leaf_epochs_20_dir), SERVER_TIME_KEY, weighted=True)

    leaf_epochs_5_size_10_acc_round = get_acc(load_data(leaf_epochs_5_size_10_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_5_size_10_acc_time = get_acc(load_data(leaf_epochs_5_size_10_dir), SERVER_TIME_KEY, weighted=True)
    leaf_epochs_10_size_10_acc_round = get_acc(load_data(leaf_epochs_10_size_10_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_10_size_10_acc_time = get_acc(load_data(leaf_epochs_10_size_10_dir), SERVER_TIME_KEY, weighted=True)
    leaf_epochs_15_acc_round = get_acc(load_data(leaf_epochs_15_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_15_acc_time = get_acc(load_data(leaf_epochs_15_dir), SERVER_TIME_KEY, weighted=True)
    leaf_epochs_20_size_10_acc_round = get_acc(load_data(leaf_epochs_20_size_10_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_20_size_10_acc_time = get_acc(load_data(leaf_epochs_20_size_10_dir), SERVER_TIME_KEY, weighted=True)

    plot_accuracy_vs_round_number_leaf_lan_aware(weighted=True)
    plot_accuracy_vs_server_time_leaf_lan_aware(weighted=True)
    print()
