import types
import copy
from keras.utils.generic_utils import has_arg
from keras.models import Sequential, Model

class BaseWrapper:
    """
    Wrapper for the hyperparamenters tuning.

    Arguments:
        build_fn: callable function
        **sk_params: model parameters & keras functions parameters

    The `build_fn` should construct, compile and return a Keras model, which
    will then be used to fit/predict.
    
    `sk_params` takes both model parameters and any model function parameters.
    
    `build_fn` should provide default values for its arguments, so that you 
    could create the estimator without passing any values to `sk_params`.

    `sk_params` could also accept parameters for any keras function parameters
    (e.g., `epochs`, `batch_size`).
    """

    def __init__(self, build_fn=None, **sk_params):

        self.build_fn = build_fn
        self.sk_params = sk_params
        self.check_params(sk_params)
        self.model = None

    def check_params(self, params):
        """
        Checks for user typos in 'params'.

        Arguments:
            params: dictionary; the parameters to be checked

        Raises:
            ValueError: if any member of `params` is not a valid argument.
        """

        # Sequential and Models functions reference: https://github.com/keras-team/keras/blob/994c4bb338ce440a27df5db48b8f5044a3e8f611/docs/structure.py
        legal_params_fns = [
            Sequential.compile,
            Sequential.fit,
            Sequential.evaluate,
            Sequential.predict,
            Sequential.train_on_batch,
            Sequential.test_on_batch,
            Sequential.predict_on_batch,
            Sequential.fit_generator,
            Sequential.evaluate_generator,
            Sequential.predict_generator,
            Model.compile,
            Model.fit,
            Model.evaluate,
            Model.predict,
            Model.train_on_batch,
            Model.test_on_batch,
            Model.predict_on_batch,
            Model.fit_generator,
            Model.evaluate_generator,
            Model.predict_generator
        ]

        if (not isinstance(self.build_fn, types.FunctionType) and (not isinstance(self.build_fn, types.MethodType))):
            raise ValueError('build_fn is not a function or method.')
        else:
            legal_params_fns.append(self.build_fn)

        for params_name in params:
            for fn in legal_params_fns:
                if has_arg(fn, params_name):
                    break
            else:
                raise ValueError('{} is not a legal parameter'.format(params_name))        

    def filter_sk_params(self, fn, override=None):
        """
        Filters 'sk_params' and returns those in fn's arguments.

        Arguments:
            fn : arbitrary function
            override: dictionary, values to override 'sk_params'

        Returns:
            res : dictionary containing variables in both 'sk_params' and fn's arguments.
        """

        override = override or {}
        res = {}
        for name, value in self.sk_params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res

    def getMetricsValues(self, history, score=None):

        metrics = {}

        # Metrics from training phase
        for m in history.history:
            metrics[m] = history.history[m][-1]
            # metrics[m] = format(history.history[m][-1],'.5f')

        #
        if score is not None:
            for i in range(len(score)):
                score_metric_name = 'test_{}'.format(self.model.metrics_names[i])
                metrics[score_metric_name] = score[i]

        return metrics

    def get_model(self):
        """
        Gets the latest model generated.

        Arguments:
            None

        Returns:
            Keras model
        """

        return self.model

    def get_params(self):
        """
        Gets parameters for this estimator.

        Arguments:
            None

        Returns:
            Dictionary of parameter names mapped to their values.
        """

        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        """
        Sets the parameters of this estimator.

        Arguments:
            **params: Dictionary of parameter names mapped to their values.

        Returns:
            self
        """

        self.check_params(params)
        self.sk_params.update(params)
        return self

    def fit_generator(self, train_generator, val_generator, **kwargs):
        """
        Constructs a new model with 'build_fn' & Train the model.

        Arguments:
            train_generator : a generator or a Sequence object for the training data.
            val_generator : a generator or a Sequence object for the validation data.
            **kwargs: dictionary arguments 
                Legal arguments are the arguments of Keras fit_generator function.

        Returns:
            history : object
                details about the training history at each epoch.
        """

        # Create the Keras models
        self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
        
        # Filter parameters for the Keras functions
        fit_args = copy.deepcopy(self.filter_sk_params(self.model.fit_generator))

        history = self.model.fit_generator(train_generator, validation_data=val_generator, **fit_args)

        return history

    def predict_generator(self, test_generator, **kwargs):
        """
        Constructs a new model with 'build_fn' & Generates predictions for the input samples from a data generator.

        Arguments:
            test_generator : a generator or a Sequence object for the testing data.
            **kwargs: dictionary arguments 
                Legal arguments are the arguments of Keras predict_generator function.

        Returns:
            predictions : Numpy array(s) of predictions.
        """

        # Create the Keras models
        self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
        
        # Filter parameters for the Keras functions
        fit_args = copy.deepcopy(self.filter_sk_params(self.model.predict_generator))

        predictions = self.model.predict_generator(test_generator, **fit_args)

        return predictions

    def evaluate_generator(self, test_generator, **kwargs):
        """
        Constructs a new model with 'build_fn' & Evaluates the model on a data generator..

        Arguments:
            test_generator : a generator or a Sequence object for the testing data.
            **kwargs: dictionary arguments 
                Legal arguments are the arguments of Keras evaluate_generator function.

        Returns:
            scores : Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).
        """

        # Create the Keras models
        self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
        
        # Filter parameters for the Keras functions
        fit_args = copy.deepcopy(self.filter_sk_params(self.model.evaluate_generator))

        scores = self.model.evaluate_generator(test_generator, **fit_args)

        return scores        