"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)

        # Download time of client from server
        server_download_time = server.get_download_time()
        # print('--- download time %d ---' % (server_download_time))

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        # sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size,
        #                                      minibatch=args.minibatch, batch_num=BATCH_NUM)

        train_time = sys_metrics["train_time_max"]
        # train_time = 0
        # print('--- train time %d ---' % (train_time))

        server_upload_time = server.get_upload_time()
        # print('--- upload time %d ---' % (server_upload_time))

        total_time = server_download_time + train_time + server_upload_time
        # print('--- net train time ratio: %.1f ---' % ((total_time-train_time)/train_time))

        server.pass_time(total_time)

        # sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)
        
        # Update server model
        server.update_model()
        # server.update_model_mom()

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
    
    # train_stat_metrics = server.test_model(clients, server_time, set_to_use='train')
    # print_metrics(train_stat_metrics, num_samples, prefix='train_')
    # writer(num_round, train_stat_metrics, 'train')

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
    to_ret = None
    f = open("acc.txt", 'a')
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        f.write('{}: {}, 10th percentile: {}, 50th percentile: {}, 90th percentile {}'.format
                (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights, axis=0),
                 np.percentile(ordered_metric, 10, axis=0),
                 np.percentile(ordered_metric, 50, axis=0),
                 np.percentile(ordered_metric, 90, axis=0)))
        f.write("\n")
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()
