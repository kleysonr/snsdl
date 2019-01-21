import itertools

class BaseModel:

    def __init__(self, params):

        self.params = params

    def create_model(self):
        raise RuntimeError('Must be implemented by subclasses.')

    def getSearchParams(self):
        """Gets the combination of all params for the model."""

        if self.params is None:
            raise ValueError('Define the space search params.')

        elif type(self.params) != dict:
            raise ValueError('Space search params must be a Dict. (e.g. {"lr": [0.01,0.001], "optimizer": ["adam","sgd"], "epochs": [100, 1000] }')

        return list(dict(zip(self.params, x)) for x in itertools.product(*self.params.values()))

