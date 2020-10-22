import random
import warnings
import numpy as np


class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data

    # def train(self, num_epochs=1, batch_size=10, minibatch=None, batch_num=1):
    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train(data, num_epochs, batch_size)
            # comp, update = self.model.train(data, num_epochs, batch_size, batch_num)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}

            # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
            num_epochs = 1
            comp, update = self.model.train(data, num_epochs, num_data)
        num_train_samples = len(data['y'])
        train_time = self.get_train_time(self.model, num_train_samples, batch_size, num_epochs)
        # train_time = self.get_train_time(self.model, num_train_samples, batch_size, num_epochs, batch_num)
        return comp, num_train_samples, update, train_time

    def get_train_time(self, model, num_sample, batch_size, num_epoch):
    # def get_train_time(self, model, num_sample, batch_size, num_epoch, batch_num):
        '''
            return the training time using look up table
            Args:
                model: device model(should be supported)
                num_sample: number of samples
                batch_size: batch size
                num_epoch: number of epoches
        '''

        reddit_mean = [2596, 916, 527.7]
        reddit_std = [0, 0, 0]
        celeba_mean = [5392, 1355, 561]
        celeba_std = [0, 0, 0]
        # celeba_std = [982.5, 54.6, 20.3]
        femnist_mean = [1642, 588, 179]
        femnist_std = [0, 0, 0]
        # femnist_std = [99.5, 23.9, 2.3]
        shakespeare_mean = [28621, 13579, 10681]    # batch size = 100
        shakespeare_std = [0, 0, 0]    # batch size = 100
        ii = 0
        # train_time_per_batch = np.random.normal(femnist_mean[ii], femnist_std[ii]) / 1000
        train_time_per_batch = np.random.normal(celeba_mean[ii], celeba_std[ii]) / 1000
        # train_time_per_batch = np.random.normal(shakespeare_mean[ii], shakespeare_std[ii]) / 1000
        # train_time_per_batch = np.random.normal(reddit_mean[ii], reddit_std[ii]) / 1000
        # ii = self.supported_devices.index(model)
        # if self.model == 'cnn' and self.dataset == 'celeba':
        #     train_time_per_batch = np.random.normal(celeba_mean[ii], celeba_std[ii]) / 1000
        # elif 'stacked_lstm' in self.model and  'reddit' in self.dataset:
        #     train_time_per_batch = np.random.normal(reddit_mean[ii], reddit_std[ii]) / 1000
        # elif self.model == 'cnn' and self.dataset == 'femnist':
        #     train_time_per_batch = np.random.normal(femnist_mean[ii], femnist_std[ii]) / 1000
        #     #print(train_time_per_batch)
        # elif 'stacked_lstm' in self.model and self.dataset == 'shakespeare':
        #     train_time_per_batch = np.random.normal(shakespeare_mean[ii], shakespeare_std[ii]) / 1000
        # else:
        #     train_time_per_batch = np.random.normal(reddit_mean[ii], reddit_std[ii]) / 1000
        # print(train_time_per_batch)
        return num_epoch * ((num_sample-1)//batch_size + 1) * train_time_per_batch
        # return num_epoch * batch_num * train_time_per_batch

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
