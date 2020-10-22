"""Helper to visualize metrics-2."""
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

    # 同构lan vs leaf， 固定lan的个数=5， 比较leaf-2,lan-10,20,30
    plt.plot(leaf_epochs_20_acc_round[NUM_ROUND_KEY], leaf_epochs_20_acc_round[ACCURACY_KEY],
             label='Leaf,E=20,WB=2',
             linestyle='-', color='red', linewidth=3)
    plt.plot(lan_epochs_2_agg_10_speed_20_acc_round[NUM_ROUND_KEY], lan_epochs_2_agg_10_speed_20_acc_round[ACCURACY_KEY],
             label='Lan,E=2,A=10,LB=20',
             linestyle=':', color='blue', linewidth=3)
    plt.plot(lan_epochs_2_agg_10_speed_10_acc_round[NUM_ROUND_KEY],
             lan_epochs_2_agg_10_speed_10_acc_round[ACCURACY_KEY],
             label='Lan,E=2,A=10,LB=10',
             linestyle='--', color='green', linewidth=3)
    plt.plot(lan_epochs_2_agg_10_speed_30_acc_round[NUM_ROUND_KEY],
             lan_epochs_2_agg_10_speed_30_acc_round[ACCURACY_KEY],
             label='Lan,E=2,A=10,LB=30',
             linestyle='-.', color='purple', linewidth=3)

    # plt.plot(leaf_epochs_20_acc_round[NUM_ROUND_KEY], leaf_epochs_20_acc_round[ACCURACY_KEY],
    #          label='Leaf,E=20',
    #          linestyle='-', color='red', linewidth=3)
    # plt.plot(lan_epochs_2_agg_10_lan_5_acc_round[NUM_ROUND_KEY], lan_epochs_2_agg_10_lan_5_acc_round[ACCURACY_KEY],
    #          label='Lan,E=2,A=10,LAN=5',
    #          linestyle=':', color='blue', linewidth=3)
    # plt.plot(lan_epochs_2_agg_10_lan_1_acc_round[NUM_ROUND_KEY], lan_epochs_2_agg_10_lan_1_acc_round[ACCURACY_KEY],
    #          label='Lan,E=2,A=10,LAN=1',
    #          linestyle='--', color='orange', linewidth=3)
    # plt.plot(lan_epochs_2_agg_10_lan_2_acc_round[NUM_ROUND_KEY], lan_epochs_2_agg_10_lan_2_acc_round[ACCURACY_KEY],
    #          label='Lan,E=2,A=10,LAN=2',
    #          linestyle='-.', color='green', linewidth=3)
    # plt.plot(lan_epochs_2_agg_10_lan_10_acc_round[NUM_ROUND_KEY], lan_epochs_2_agg_10_lan_10_acc_round[ACCURACY_KEY],
    #          label='Lan,E=2,A=10,LAN=10',
    #          linestyle=':', color='purple', linewidth=3)


    plt.legend(loc='lower right')
    plt.grid(True)

    # plt.xlim(0,80)
    plt.ylabel('Weighted Accuracy')
    plt.xlabel('Round Number')
    _set_plot_properties(kwargs)
    # plt.savefig('pdf/eval-acc-round-speed.pdf', bbox_inches='tight')
    plt.show()


def plot_accuracy_vs_server_time_leaf_lan_aware(weighted=False, plot_stds=False, figsize=(10, 8), title_fontsize=16, **kwargs):
    """Plots the clients' average test accuracy vs. the round number.
    Args:
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        title_fontsize: Font size for the plot's title.
        kwargs: Arguments to be passed to _set_plot_properties."""
    plt.figure(figsize=figsize)
    # title_weighted = 'Weighted' if weighted else 'Unweighted'
    # plt.title('Weighted Accuracy vs Clock Time', fontsize=title_fontsize)
    # fig, ax = plt.subplots(1, 1, figsize=(6,4))
    round = 16

    # 同构lan vs leaf， 固定lan的个数=5， 比较leaf-2,lan-10,20,30
    # plt.plot(get_time_round(leaf_epochs_20_acc_time[SERVER_TIME_KEY], round), leaf_epochs_20_acc_time[ACCURACY_KEY][0:round],
    #          label='Leaf,E=20,WB=2',
    #          linestyle='-', color='red', linewidth=3)
    # plt.plot(get_time_round(lan_epochs_2_agg_10_speed_20_acc_time[SERVER_TIME_KEY], round), lan_epochs_2_agg_10_speed_20_acc_time[ACCURACY_KEY][0:round],
    #          label='Lan,E=2,A=10,LB=20',
    #          linestyle=':', color='blue', linewidth=3)
    # plt.plot(get_time_round(lan_epochs_2_agg_10_speed_10_acc_time[SERVER_TIME_KEY], round), lan_epochs_2_agg_10_speed_10_acc_time[ACCURACY_KEY][0:round],
    #          label='Lan,E=2,A=10,LB=10',
    #          linestyle='--', color='green', linewidth=3)
    # plt.plot(get_time_round(lan_epochs_2_agg_10_speed_30_acc_time[SERVER_TIME_KEY], round), lan_epochs_2_agg_10_speed_30_acc_time[ACCURACY_KEY][0:round],
    #          label='Lan,E=2,A=10,LB=30',
    #          linestyle='-.', color='purple', linewidth=3)

    plt.plot([time_second/3600 for time_second in leaf_epochs_20_acc_time[SERVER_TIME_KEY]], leaf_epochs_20_acc_time[ACCURACY_KEY],
             label='WAN-FL,BW=2',
             linestyle='-', color='red', linewidth=1)
    plt.plot([time_second/3600 for time_second in lan_ring_1_acc_time[SERVER_TIME_KEY]], lan_ring_1_acc_time[ACCURACY_KEY],
             label='LanFL(Ring-1),BL=20',
             linestyle='--', color='orange', linewidth=1)
    plt.plot([time_second/3600 for time_second in lan_ring_4_acc_time[SERVER_TIME_KEY]], lan_ring_4_acc_time[ACCURACY_KEY],
             label='LanFL(Ring-4),BL=62',
             linestyle='--', color='green', linewidth=1)
    plt.plot([time_second/3600 for time_second in lan_ps_1_acc_time[SERVER_TIME_KEY]], lan_ps_1_acc_time[ACCURACY_KEY],
             label='LanFL(PS-1),BL=22',
             linestyle='-.', color='blue', linewidth=1)
    plt.plot([time_second/3600 for time_second in lan_ps_4_acc_time[SERVER_TIME_KEY]], lan_ps_4_acc_time[ACCURACY_KEY],
             label='LanFL(PS-4),BL=16',
             linestyle='-.', color='purple', linewidth=1)

    plt.legend(loc='lower right', fontsize=30)
    plt.grid(True)

    plt.xlim(0,30)
    plt.ylabel('Weighted Accuracy(%)', fontsize=26)
    plt.xlabel('Clock Time(Hour)', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    _set_plot_properties(kwargs)
    plt.savefig('pdf/eval-time-ps-ring-femnist.pdf', bbox_inches='tight')
    plt.show()


def get_x_y(list_to_simple):
    x = [list_to_simple[i] for i in range(0, len(list_to_simple), 10)]
    return x


def get_time_round(time_second_list, round):
    time_hour_list = [time_second/3600 for time_second in time_second_list]
    return time_hour_list[0:round]


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
    leaf_epochs_20_dir = 'ps-ring/leaf/clients_40/epochs_10/metrics/metrics_stat.csv'
    lan_ps_1_dir = 'ps-ring/lan/clients_5×8/epochs_2_agg_5/ps-1/metrics/metrics_stat.csv'
    lan_ps_4_dir = 'ps-ring/lan/clients_5×8/epochs_2_agg_5/ps-4/metrics/metrics_stat.csv'
    lan_ring_1_dir = 'ps-ring/lan/clients_5×8/epochs_2_agg_5/ring-1/metrics/metrics_stat.csv'
    lan_ring_4_dir = 'ps-ring/lan/clients_5×8/epochs_2_agg_5/ring-4/metrics/metrics_stat.csv'

    # get accuracy and std
    leaf_epochs_20_acc_round = get_acc(load_data(leaf_epochs_20_dir), NUM_ROUND_KEY, weighted=True)
    leaf_epochs_20_acc_time = get_acc(load_data(leaf_epochs_20_dir), SERVER_TIME_KEY, weighted=True)

    lan_ps_1_acc_round = get_acc(load_data(lan_ps_1_dir), NUM_ROUND_KEY, weighted=True)
    lan_ps_1_acc_time = get_acc(load_data(lan_ps_1_dir), SERVER_TIME_KEY, weighted=True)
    lan_ps_4_acc_round = get_acc(load_data(lan_ps_4_dir), NUM_ROUND_KEY, weighted=True)
    lan_ps_4_acc_time = get_acc(load_data(lan_ps_4_dir), SERVER_TIME_KEY, weighted=True)
    lan_ring_1_acc_round = get_acc(load_data(lan_ring_1_dir), NUM_ROUND_KEY, weighted=True)
    lan_ring_1_acc_time = get_acc(load_data(lan_ring_1_dir), SERVER_TIME_KEY, weighted=True)
    lan_ring_4_acc_round = get_acc(load_data(lan_ring_4_dir), NUM_ROUND_KEY, weighted=True)
    lan_ring_4_acc_time = get_acc(load_data(lan_ring_4_dir), SERVER_TIME_KEY, weighted=True)

    # plot_accuracy_vs_round_number_leaf_lan_aware(weighted=True)
    plot_accuracy_vs_server_time_leaf_lan_aware(weighted=True)
    print()
