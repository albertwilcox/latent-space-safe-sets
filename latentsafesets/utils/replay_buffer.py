import numpy as np


class ReplayBuffer:
    """
    This replay buffer uses numpy to efficiently store arbitrary data. Keys can be whatever,
    but once you push data to the buffer new data must all have the same keys (to keep parallel
    arrays parallel).
    """
    def __init__(self, size=10000):
        self.size = size

        self.data = {}
        self._index = 0
        self._len = 0

    def store_transitions(self, transistions):
        """
        Stores transitions
        :param transistions: a list of dictionaries encoding transitions. Keys can be anything
        """
        assert transistions[-1]['done'] > 0, "Last transition must be end of trajectory"
        for transition in transistions:
            self.store_transition(transition)

    def store_transition(self, transition):
        if len(self.data) > 0:
            key_set = set(self.data)
        else:
            key_set = set(transition)

        assert key_set == set(transition), "Expected transition to have keys %s" % key_set

        for key in key_set:
            data = self.data.get(key, None)
            new_data = np.array(transition[key])
            if data is None:
                data = np.zeros((self.size, *new_data.shape), dtype=new_data.dtype)
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

        return self._im_to_float({key: self.data[key][indices] for key in self.data})

    def sample_positive(self, batch_size, key, ensemble=0):
        """
        Samples only from the entries where the array corresponding to key is nonzero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        nonzeros = self.data[key].nonzero()[0]
        if ensemble == 0:
            indices = np.random.randint(len(nonzeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(nonzeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return self._im_to_float({key: self.data[key][nonzeros[indices]] for key in self.data})

    def sample_negative(self, batch_size, key, ensemble=0):
        """
        Samples only from the entries where the array corresponding to key is zero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        zeros = (1 - self.data[key]).nonzero()[0]
        # print(nonzeros)
        if ensemble == 0:
            indices = np.random.randint(len(zeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(zeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return self._im_to_float({key: self.data[key][zeros[indices]] for key in self.data})

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
            out = self.data[key][idxs]
            out_dict[key] = out
        return self._im_to_float(out_dict)

    def all_transitions(self):
        for i in range(len(self)):
            transition = {key: self.data[key][i] for key in self.data}
            yield transition

    def _im_to_float(self, out_dict):
        for key in out_dict:
            if key in ('obs', 'next_obs'):
                out_dict[key] = out_dict[key] / 255
        return out_dict

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
