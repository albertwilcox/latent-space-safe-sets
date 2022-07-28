from abc import ABC, abstractmethod#abstract classes


class Trainer(ABC):

    @abstractmethod
    def initial_train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError
