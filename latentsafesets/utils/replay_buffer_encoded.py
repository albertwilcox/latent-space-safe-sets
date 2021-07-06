import numpy as np
import latentsafesets.utils.pytorch_utils as ptu


class EncodedReplayBuffer:
    """
    This replay buffer uses numpy to efficiently store arbitrary data. Keys can be whatever,
    but once you push data to the buffer new data must all have the same keys (to keep parallel
    arrays parallel).

    This replay buffer replaces all images with their representation from encoder
    """
    def __init__(self, encoder, size=10000):
        self.size = size
        self.encoder = encoder

        self.data = {}
        self._index = 0
        self._len = 0
        self.im_keys = ('obs', 'next_obs')

    def store_transitions(self, transitions):
        """
        Stores transitions
        :param transitions: a list of dictionaries encoding transitions. Keys can be anything
        """
        assert transitions[-1]['done'] > 0, "Last transition must be end of trajectory"
        for transition in transitions:
            self.store_transition(transition)

    def store_transition(self, transition):
        if len(self.data) > 0:
            key_set = set(self.data)
        else:
            key_set = set(transition)

        # assert key_set == set(transition), "Expected transition to have keys %s" % key_set

        for key in key_set:
            data = self.data.get(key, None)

            new_data = np.array(transition[key])
            if key in self.im_keys:
                im = np.array(transition[key])
                im = ptu.torchify(im)
                new_data_mean, new_data_log_std = self.encoder(im[None] / 255)
                new_data_mean = new_data_mean.squeeze().detach().cpu().numpy()
                new_data_log_std = new_data_log_std.squeeze().detach().cpu().numpy()
                new_data = np.dstack((new_data_mean, new_data_log_std)).squeeze()

            if data is None:
                data = np.zeros((self.size, *new_data.shape))
            data[self._index] = new_data
            self.data[key] = data

        self._index = (self._index + 1) % self.size
        self._len = min(self._len + 1, self.size)

    def sample(self, batch_size, ensemble=0):
        if ensemble == 0:
            indices = np.random.randint(len(self), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(self), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, indices) for key in self.data}

    def sample_positive(self, batch_size, key, ensemble=0):
        """
        Samples only from the entries where the array corresponding to key is nonzero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        nonzeros = self.data[key].nonzero()[0]
        # print(nonzeros)
        if ensemble == 0:
            indices = np.random.randint(len(nonzeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(nonzeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, nonzeros[indices]) for key in self.data}

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample_chunk(self, batch_size, length, ensemble=0):
        if ensemble == 0:
            idxs = np.asarray([self._sample_idx(length) for _ in range(batch_size)])
        elif ensemble > 0:
            idxs = np.asarray([[self._sample_idx(length) for _ in range(batch_size)]
                               for _ in range(ensemble)])
        else:
            raise ValueError("ensemble size cannot be negative")
        out_dict = {}
        for key in self.data:
            out = self._extract(key, idxs)
            out_dict[key] = out
        return out_dict

    def all_transitions(self):
        for i in range(len(self)):
            transition = {key: self.data[key] for key in self.data}
            yield transition

    def _extract(self, key, indices):
        if key in self.im_keys:
            dat = self.data[key][indices]
            dat_mean, dat_log_std = np.split(dat, 2, axis=-1)
            dat_std = np.exp(dat_log_std)
            return np.random.normal(dat_mean.squeeze(), dat_std.squeeze())
        else:
            return self.data[key][indices]

    def _sample_idx(self, length):
        valid_idx = False
        idxs = None
        while not valid_idx:
            idx = np.random.randint(0, len(self) - length)
            idxs = np.arange(idx, idx + length) % self.size
            # Make sure data does not cross the memory index
            valid_idx = self._index not in idxs[1:]
            if 'done' in self.data:
                end = np.any(self.data['done'][idxs[:-1]])
                valid_idx = valid_idx and not end
        return idxs

    def __len__(self):
        return self._len
