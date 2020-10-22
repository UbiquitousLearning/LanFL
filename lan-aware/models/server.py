import numpy as np
import json

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server:
    speed_lan = None
    try:
        with open('speed_lan.json', 'r') as f:
            speed_lan = json.load(f)
    except FileNotFoundError as e:
        speed_lan = None
    speed_wan = None
    try:
        with open('speed_wan.json', 'r') as f:
            speed_wan = json.load(f)
    except FileNotFoundError as e:
        speed_wan = None
        # logger.warn('no central\'s network speed trace was found, set all communication time to 0.0s')

    def __init__(self, client_model):
        self._cur_time = 0
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.updates = []
        self.upload_speed_u_wan = Server.speed_wan["up_u"]
        self.upload_speed_sigma_wan = Server.speed_wan["up_sigma"]
        self.download_speed_u_wan = Server.speed_wan["down_u"]
        self.download_speed_sigma_wan = Server.speed_wan["down_sigma"]

        self.speed_lan = Server.speed_lan

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def get_download_time_wan(self):
        download_speed = np.random.normal(self.download_speed_u_wan, self.download_speed_sigma_wan)
        while download_speed < 0:
            download_speed = np.random.normal(self.download_speed_u_wan, self.download_speed_sigma_wan)
        download_time = self.client_model.size * 8 / download_speed / 1024 / 1024
        return float(download_time)

    def get_upload_time_wan(self):
        upload_speed = np.random.normal(self.upload_speed_u_wan, self.upload_speed_sigma_wan)
        while upload_speed < 0:
            upload_speed = np.random.normal(self.upload_speed_u_wan, self.upload_speed_sigma_wan)
        upload_time = self.client_model.size * 8 / upload_speed / 1024 / 1024
        return float(upload_time)

    def get_download_time_lan(self, server_local_index):
        download_speed_u = self.speed_lan[str(server_local_index)]["down_u"]
        download_speed_sigma = self.speed_lan[str(server_local_index)]["down_sigma"]
        download_speed = np.random.normal(download_speed_u, download_speed_sigma)
        while download_speed < 0:
            download_speed = np.random.normal(download_speed_u, download_speed_sigma)
        download_time = self.client_model.size * 8 / download_speed / 1024 / 1024
        return float(download_time)

    def get_upload_time_lan(self, server_local_index):
        upload_speed_u = self.speed_lan[str(server_local_index)]["up_u"]
        upload_speed_sigma = self.speed_lan[str(server_local_index)]["up_sigma"]
        upload_speed = np.random.normal(upload_speed_u, upload_speed_sigma)
        while upload_speed < 0:
            upload_speed = np.random.normal(upload_speed_u, upload_speed_sigma)
        upload_time = self.client_model.size * 8 / upload_speed / 1024 / 1024
        return float(upload_time)

    def get_cur_time(self):
        return self._cur_time

    def pass_time(self, sec):
        self._cur_time += sec

    # def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, batch_num=1):
    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.

        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        train_time_list = []
        for c in clients:
            c.model.set_params(self.model)
            comp, num_samples, update, train_time = c.train(num_epochs, batch_size, minibatch)
            # comp, num_samples, update, train_time = c.train(num_epochs, batch_size, minibatch, batch_num)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
            sys_metrics[c.id]["train_time"] = train_time
            train_time_list.append(train_time)

            self.updates.append((num_samples, update))
        train_time_max = max(train_time_list)
        sys_metrics["train_time_max"] = train_time_max

        return sys_metrics

    def update_lan_model(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []
        return total_weight

    def update_wan_model(self, lan_model_list):
        total_weight = 0.
        base = [0] * len(lan_model_list[0][1])
        for (client_samples, client_model) in lan_model_list:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln

    def test_model(self, clients_to_test, server_time, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
            metrics[client.id]['server_time'] = server_time

        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess = self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()