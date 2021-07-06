from abc import ABC, abstractmethod


class Trainer(ABC):

    @abstractmethod
    def initial_train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError
