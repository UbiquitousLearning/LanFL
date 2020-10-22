"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, NUM_LAN, NUM_AGG_ROUNDS
from client import Client
from server import Server
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# tune parameter by batch_num
# BATCH_NUM = 20

def main():

    args = parse_args()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]

    group_clients_strategy = 'Equal_Division'

    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)
    print('model size: %d' % client_model.size)

    # Create server
    server = Server(client_model)

    # Create clients
    clients = setup_clients(args.dataset, client_model, args.use_val_set)
    # clients = clients[0:100]
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # Group clients
    clients_group_lan = group_clients_for_server_local(NUM_LAN, group_clients_strategy, clients=clients)

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set, server.get_cur_time())

    # Simulate training
    for i in range(num_rounds):
        server_cur_time = server.get_cur_time()
        print('--- current time %d ---' % (server_cur_time))
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Download time of client from server
        server_download_time = server.get_download_time_wan()
        # print('--- download time %d ---' % (server_download_time))

        lan_model_list = [] # for wan server to aggregate
        lan_train_time_list = []
        train_time_init_list = []
        for j in range(NUM_LAN):
            client_model.set_params(server.model)
            model_test = client_model.get_params()
            server_lan = Server(client_model)
            server_total_samples = 0
            lan_train_time = 0
            train_time_init = 0
            for k in range(NUM_AGG_ROUNDS):
                client_model.set_params(server_lan.model)
                model_test = client_model.get_params()
                server_lan = Server(client_model)

                # Select clients to train this round
                server_lan.select_clients(i, online(clients_group_lan[j]), num_clients=clients_per_round)
                c_ids, c_groups, c_num_samples = server_lan.get_clients_info(server_lan.selected_clients)

                # Simulate server model training on selected clients' data
                # sys_metrics = server_lan.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch, batch_num=BATCH_NUM)
                sys_metrics = server_lan.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)

                train_time = sys_metrics["train_time_max"]
                # train_time = 0
                train_time_init = train_time_init + train_time

                # Upload model of each client to a local server in a lan
                lan_upload_time = server_lan.get_upload_time_lan(j+1)

                # aggregate the model in a lan
                server_total_samples += server_lan.update_lan_model()

                # Download the new model to each client in a lan
                lan_download_time = server_lan.get_download_time_lan(j+1)

                if k == NUM_AGG_ROUNDS - 1:
                    lan_train_time += (train_time + lan_upload_time)
                else:
                    lan_train_time += (train_time + lan_upload_time + lan_download_time)

            train_time_init_list.append(train_time_init)
            lan_train_time_list.append(lan_train_time)
            lan_model_list.append((server_total_samples, server_lan.model))

        actual_train_time_init = max(train_time_init_list)
        actual_lan_time = max(lan_train_time_list)
        server_upload_time = server.get_upload_time_wan()
        total_time = server_download_time + actual_lan_time + server_upload_time

        server.pass_time(total_time)

        # sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

        # Update wan server model
        server.update_wan_model(lan_model_list)

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set, server.get_cur_time())

    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
    print('Model saved in path: %s' % save_path)

    # Close models
    server.close_model()


def online(clients):
    """We assume all users are always online."""
    return clients


def group_clients_for_server_local(num_lan_local, group_clients_strategy, clients=None):
    """
    将全部的client按照local server进行分组
    :param cfg:
    :param clients:
    :return:
    """
    if group_clients_strategy == 'Equal_Division':
        num_clients_per_group = int(len(clients) / num_lan_local)
        list_temp = []
        for i in range(0, len(clients), num_clients_per_group):
            list_temp.append(clients[i:i + num_clients_per_group])
        return list_temp


def select_client_for_train(clients, clients_per_round):
    np.random.seed()
    selected_clients = np.random.choice(clients, clients_per_round, replace=False)
    return [(c.num_train_samples, c.num_test_samples) for c in selected_clients]


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, model=None, use_val_set=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model)

    return clients


def get_stat_writer_function(ids, num_samples, args):

    def writer_fn(num_round, metrics, train_or_test):
        metrics_writer.print_metrics(
            num_round, ids, metrics, num_samples, train_or_test, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(
    num_round, server, clients, num_samples, args, writer, use_val_set, server_time):

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, server_time, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    # metric_names = metric_names[0:2]
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()
