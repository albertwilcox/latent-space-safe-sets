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

        self.data = {}#finally it becomes a dict where each key's value have size number of values
        self._index = 0
        self._len = 0
        self.im_keys = ('obs', 'next_obs')

    def store_transitions(self, transitions):#transitions is 1 traj having 100 steps
        """
        Stores transitions
        :param transitions: a list of dictionaries encoding transitions. Keys can be anything
        """
        assert transitions[-1]['done'] > 0, "Last transition must be end of trajectory"
        for transition in transitions:#a transition is 1 step#It is a dictionary
            self.store_transition(transition)

    def store_transition(self, transition):#a transition is 1 step#It is a dictionary
        if len(self.data) > 0:#at first it is not like this, from second it is like this
            key_set = set(self.data)#the keys of self.data are the keys of transition!
        else:#at first it is like this
            key_set = set(transition)#you only get the keys of that dictionary! python usage!

        # assert key_set == set(transition), "Expected transition to have keys %s" % key_set

        for key in key_set:#it is a set
            data = self.data.get(key, None)#.get() is to get the value of a key

            new_data = np.array(transition[key])#it seems already converts list to array
            if key in self.im_keys:
                im = np.array(transition[key])#seems to be the image?
                im = ptu.torchify(im)
                new_data_mean, new_data_log_std = self.encoder(im[None] / 255)
                new_data_mean = new_data_mean.squeeze().detach().cpu().numpy()
                new_data_log_std = new_data_log_std.squeeze().detach().cpu().numpy()
                new_data = np.dstack((new_data_mean, new_data_log_std)).squeeze()

            if data is None:
                data = np.zeros((self.size, *new_data.shape))#then fill one by one
            data[self._index] = new_data#now fill one by one
            self.data[key] = data#the value of self.data[key] is a np array#the way to init a value in a dict

        self._index = (self._index + 1) % self.size#no more no less, just 10000 piece of data#a queue!
        self._len = min(self._len + 1, self.size)#a thing saturate at self.size!
        #I think I have understood the above function!
    def sample(self, batch_size, ensemble=0):#bs=256 by default
        if ensemble == 0:
            indices = np.random.randint(len(self), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(self), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, indices) for key in self.data}#106

    def sample_positive(self, batch_size, key, ensemble=0):
        """
        Samples only from the entries where the array corresponding to key is nonzero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        nonzeros = self.data[key].nonzero()[0]#self.data[key] is the value#get the safe ones!
        # print(nonzeros)
        if ensemble == 0:
            indices = np.random.randint(len(nonzeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(nonzeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, nonzeros[indices]) for key in self.data}#106

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
            out = self._extract(key, idxs)#106
            out_dict[key] = out
        return out_dict

    def all_transitions(self):
        for i in range(len(self)):
            transition = {key: self.data[key] for key in self.data}
            yield transition

    def _extract(self, key, indices):#give the term you want, then return the value
        if key in self.im_keys:#obs and next_obs
            dat = self.data[key][indices]
            dat_mean, dat_log_std = np.split(dat, 2, axis=-1)
            dat_std = np.exp(dat_log_std)
            return np.random.normal(dat_mean.squeeze(), dat_std.squeeze())
        else:#if it is not an image, then just return the value
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
