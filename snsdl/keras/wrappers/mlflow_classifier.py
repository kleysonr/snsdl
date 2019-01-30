import os
import copy
import mlflow
import pandas as pd
from keras.callbacks import CSVLogger
from snsdl.keras.wrappers import BaseWrapper

class MlflowClassifier(BaseWrapper):
    """ Implementation of the mlflow classifier API for Keras using generators."""

    def __init__(self, build_fn, train_generator, test_generator, val_generator=None, tracking_server=None, artifacts_dir=None, **sk_params):

        _sk_params = copy.deepcopy(sk_params)

        self.artifacts_dir = artifacts_dir

        if artifacts_dir is not None:

            # Logs training logs in a csv file
            try:
                _sk_params['callbacks']
            except KeyError:
                _sk_params['callbacks'] = []
            finally:
                os.makedirs(os.path.join(artifacts_dir, 'text'), exist_ok=True)
                _sk_params['callbacks'].append(CSVLogger(os.path.join(artifacts_dir, 'text', 'training_log.csv')))

        # Initialize superclass parameters first
        super(MlflowClassifier, self).__init__(build_fn, train_generator, test_generator, val_generator, **_sk_params)

        # We don't want to force people to have tracking server
        # running on localhost as it tracks in mlruns directory
        if tracking_server is not None:
            # Tracking URI
            if not tracking_server.startswith("http"):
                mlflow_tracking_uri = 'http://' + tracking_server + ':5000'
            else:
                mlflow_tracking_uri = tracking_server
            # Set the Tracking URI
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            print("MLflow Tracking URI: %s" % mlflow_tracking_uri)
        else:
            print("MLflow Tracking URI: %s" % "local directory 'mlruns'")

    def log(self, experiment_id=None):

        if experiment_id is not None:
            mlflow.set_experiment(experiment_id)

        training_log = None
        if os.path.isfile(os.path.join(self.artifacts_dir, 'text', 'training_log.csv')):
            training_log = pd.read_csv(os.path.join(self.artifacts_dir, 'text', 'training_log.csv'))

            # Create a pretty training log file
            f = open(os.path.join(self.artifacts_dir, 'text', 'training_log.txt'), 'w')
            f.write(training_log.to_string())
            f.close()

            # training_log = training_log.to_dict()

        with mlflow.start_run():

            # print out current run_uuid
            run_uuid = mlflow.active_run().info.run_uuid
            print("MLflow Run ID: %s" % run_uuid)

            # log parameters
            params = self.get_params()
            for k, v in params.items():
                if k not in ['build_fn', 'callbacks']:
                    mlflow.log_param(k, v)

            # Log artifacts
            if self.artifacts_dir is not None:
                mlflow.log_artifacts(self.artifacts_dir, "results")

            # Get the training, validation and testing metrics
            metrics = self.getMetricsValues()

            # Log metrics
            if metrics is not None:
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

            # log model
            # mlflow.keras.log_model(self.get_model(), "models")

            # # save model locally
            # pathdir = "keras_models/" + run_uuid
            # model_dir = self.get_directory_path(pathdir, False)
            # ktrain_cls.keras_save_model(model, model_dir)

            # # Write out TensorFlow events as a run artifact
            # print("Uploading TensorFlow events as a run artifact.")
            # mlflow.log_artifacts(output_dir, artifact_path="events")

            mlflow.end_run()