"""Helper to visualize metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from decimal import Decimal

models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(models_dir)

from baseline_constants import (
    ACCURACY_KEY,
    BYTES_READ_KEY,
    BYTES_WRITTEN_KEY,
    CLIENT_ID_KEY,
    LOCAL_COMPUTATIONS_KEY,
    NUM_ROUND_KEY,
    NUM_SAMPLES_KEY)


# def load_data(stat_metrics_file='stat_metrics.csv', sys_metrics_file='sys_metrics.csv'):
def load_data(stat_metrics_file='stat_metrics.csv'):
    """Loads the data from the given stat_metric and sys_metric files."""
    stat_metrics = pd.read_csv(stat_metrics_file) if stat_metrics_file else None
    # sys_metrics = pd.read_csv(sys_metrics_file) if sys_metrics_file else None

    if stat_metrics is not None:
        stat_metrics.sort_values(by=NUM_ROUND_KEY, inplace=True)
    # if sys_metrics is not None:
    #     sys_metrics.sort_values(by=NUM_ROUND_KEY, inplace=True)

    # return stat_metrics, sys_metrics
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


def plot_accuracy_vs_round_number(stat_metrics, weighted=False, plot_stds=False,
        figsize=(10, 8), title_fontsize=16, **kwargs):
    """Plots the clients' average test accuracy vs. the round number.

    Args:
        stat_metrics: pd.DataFrame as written by writer.py.
        weighted: Whether the average across clients should be weighted by number of
            test samples.
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        title_fontsize: Font size for the plot's title.
        kwargs: Arguments to be passed to _set_plot_properties."""
    plt.figure(figsize=figsize)
    title_weighted = 'Weighted' if weighted else 'Unweighted'
    plt.title('Accuracy vs Round Number (%s)' % title_weighted, fontsize=title_fontsize)
    for i in range(5):
        stat_metrics_lan = stat_metrics[stat_metrics['lan_index']==i]
        accuracies = stat_metrics_lan.groupby(NUM_ROUND_KEY).apply(_weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
        accuracies = accuracies.reset_index(name=ACCURACY_KEY)
        plt.plot(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY], label='lan_'+str(i))

    plt.legend(loc='down right')

    plt.ylabel('Accuracy')
    plt.xlabel('Round Number')
    _set_plot_properties(kwargs)
    plt.show()


def plot_accuracy_vs_round_number_wan_lan(lan_metrics, stat_metrics, weighted=False, plot_stds=False,
        figsize=(10, 8), title_fontsize=16, **kwargs):
    plt.figure(figsize=figsize)
    title_weighted = 'Weighted' if weighted else 'Unweighted'
    plt.title('Accuracy vs Round Number (%s)' % title_weighted, fontsize=title_fontsize)
    for i in range(5):
        stat_metrics_lan = lan_metrics[lan_metrics['lan_index']==i]
        accuracies = stat_metrics_lan.groupby(NUM_ROUND_KEY).apply(_weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
        accuracies = accuracies.reset_index(name=ACCURACY_KEY)
        plt.plot(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY], label='lan_'+str(i))
    accuracies_wan = stat_metrics.groupby(NUM_ROUND_KEY).apply(_weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
    accuracies_wan = accuracies_wan.reset_index(name=ACCURACY_KEY)
    plt.plot(accuracies_wan[NUM_ROUND_KEY], accuracies_wan[ACCURACY_KEY], label='wan')

    plt.legend(loc='lower right')

    plt.ylabel('Accuracy')
    plt.xlabel('Round Number')
    _set_plot_properties(kwargs)
    plt.show()


def plot_accuracy_vs_server_time(stat_metrics, weighted=False, plot_stds=False,
        figsize=(10, 8), title_fontsize=16, **kwargs):
    """Plots the clients' average test accuracy vs. the round number.

    Args:
        stat_metrics: pd.DataFrame as written by writer.py.
        weighted: Whether the average across clients should be weighted by number of
            test samples.
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        title_fontsize: Font size for the plot's title.
        kwargs: Arguments to be passed to _set_plot_properties."""
    plt.figure(figsize=figsize)
    title_weighted = 'Weighted' if weighted else 'Unweighted'
    plt.title('Accuracy vs Server Time (%s)' % title_weighted, fontsize=title_fontsize)
    if weighted:
        accuracies = stat_metrics.groupby('server_time').apply(_weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
        accuracies = accuracies.reset_index(name=ACCURACY_KEY)

        stds = stat_metrics.groupby('server_time').apply(_weighted_std, ACCURACY_KEY, NUM_SAMPLES_KEY)
        stds = stds.reset_index(name=ACCURACY_KEY)
    else:
        accuracies = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).mean()
        stds = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).std()

    if plot_stds:
        plt.errorbar(accuracies['server_time'], accuracies[ACCURACY_KEY], stds[ACCURACY_KEY])
    else:
        plt.plot(accuracies['server_time'], accuracies[ACCURACY_KEY])

    percentile = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False)
    percentile_10_acc_y = percentile['accuracy'].quantile(0.1)
    percentile_10_acc_x = percentile[NUM_ROUND_KEY]
    percentile_90_acc_y = percentile['accuracy'].quantile(0.9)
    percentile_90_acc_x = percentile[NUM_ROUND_KEY]

    # plt.plot(percentile_10_acc_y, linestyle=':')
    # plt.plot(percentile_90_acc_y, linestyle=':')

    # percentile_10 = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.1)
    # percentile_90 = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.9)

    # plt.plot(percentile_10[NUM_ROUND_KEY], percentile_10[ACCURACY_KEY], linestyle=':')
    # plt.plot(percentile_90[NUM_ROUND_KEY], percentile_90[ACCURACY_KEY], linestyle=':')

    # plt.legend(['Mean', '10th percentile', '90th percentile'], loc='upper left')
    plt.legend('mean', loc='upper left')

    plt.ylabel('Accuracy')
    plt.xlabel('Server Time')
    _set_plot_properties(kwargs)
    plt.show()


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
    lan_metrics = load_data('metrics_lan.csv')
    stat_metrics = load_data('metrics_stat.csv')
    # plot_accuracy_vs_round_number(lan_metrics, weighted=True)
    plot_accuracy_vs_round_number_wan_lan(lan_metrics, stat_metrics, weighted=True)
    # plot_accuracy_vs_server_time(lan_metrics, weighted=True)
    print()
