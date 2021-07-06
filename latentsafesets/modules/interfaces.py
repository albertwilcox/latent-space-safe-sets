from abc import ABC


class EncodedModule(ABC):
    """
    The purpose of this class is to make it easy to store encoders in torch modules without
    them showing up in the state dict (which breaks the project design in many ways)
    """
    def __init__(self, encoder):
        self._encoder = [encoder]

    @property
    def encoder(self):
        return self._encoder[0]
