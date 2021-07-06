from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def act(self, obs):
        raise NotImplementedError("Implement in subclass")



