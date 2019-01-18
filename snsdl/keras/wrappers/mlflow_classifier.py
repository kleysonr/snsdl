import mlflow
from snsdl.keras.wrappers import BaseWrapper

class MlflowClassifier(BaseWrapper):
    """ Implementation of the mlflow classifier API for Keras using generators."""

    def __init__(self, build_fn, tracking_server=None, **sk_params):

        # Initialize superclass parameters first
        super(MlflowClassifier, self).__init__(build_fn, **sk_params)

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

    def log(self):

        # mlflow.set_experiment("Keras_IMDB_Classifier")

        with mlflow.start_run():

            # print out current run_uuid
            run_uuid = mlflow.active_run().info.run_uuid
            print("MLflow Run ID: %s" % run_uuid)

            # log parameters
            params = self.get_params()
            for k, v in params.items():
                if (k != 'build_fn'):
                    mlflow.log_param(k, v)

            # # calculate metrics
            # binary_loss = ktrain_cls.get_binary_loss(history)
            # binary_acc = ktrain_cls.get_binary_acc(history)
            # validation_loss = ktrain_cls.get_validation_loss(history)
            # validation_acc = ktrain_cls.get_validation_acc(history)
            # average_loss = results[0]
            # average_acc = results[1]

            # # log metrics
            # mlflow.log_metric("binary_loss", binary_loss)
            # mlflow.log_metric("binary_acc", binary_acc)
            # mlflow.log_metric("validation_loss", validation_loss)
            # mlflow.log_metric("validation_acc", validation_acc)
            # mlflow.log_metric("average_loss", average_loss)
            # mlflow.log_metric("average_acc", average_acc)

            # # log artifacts
            # mlflow.log_artifacts(image_dir, "images")

            # # log model
            # mlflow.keras.log_model(model, "models")

            # # save model locally
            # pathdir = "keras_models/" + run_uuid
            # model_dir = self.get_directory_path(pathdir, False)
            # ktrain_cls.keras_save_model(model, model_dir)

            # # Write out TensorFlow events as a run artifact
            # print("Uploading TensorFlow events as a run artifact.")
            # mlflow.log_artifacts(output_dir, artifact_path="events")

            mlflow.end_run()