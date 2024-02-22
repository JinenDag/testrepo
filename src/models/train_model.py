import os
from pathlib import Path

import mlflow
import yaml

from ultralytics import YOLO

from utils import save_metrics_and_params, save_model




root_dir = Path(__file__).resolve().parents[1]  # root directory absolute path
data_dir = "HMI_data"
data_yaml_path = os.path.join(data_dir, "data.yaml")
metrics_path = os.path.join(root_dir, 'reports/train_metrics.json')


if __name__ == '__main__':

    # load the configuration file
    with open(r"params.yaml") as f:
        params = yaml.safe_load(f)

    # set the tracking uri
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # start mlflow experiment
    with mlflow.start_run(run_name=params['name']):
        # load a pre-trained model
        pre_trained_model = YOLO(params['model_type'])

        # train
        model = pre_trained_model.train(
            data=data_yaml_path,
            imgsz=params['imgsz'],
            batch=params['batch'],
            epochs=params['epochs'],
            optimizer=params['optimizer'],
            lr0=params['lr0'],
            seed=params['seed'],
            pretrained=params['pretrained'],
            name=params['name']
        )



        # log params with mlflow
        mlflow.log_param('model_type', params['model_type'])
        mlflow.log_param('epochs',params['epochs'])
        mlflow.log_param('optimizer', params['optimizer'])
        mlflow.log_param('learning_rate', params['lr0'])


        # save model
        save_model(experiment_name=params['name'])
        experiment_name=params['name']

        # save metrics csv file and training params
        save_metrics_and_params(experiment_name=params['name'])
        if os.path.isdir('runs'):
            model_weights = experiment_name + "/weights/best.pt"
            path_model_weights = os.path.join( "./runs/detect", model_weights)
            path_model_metrics = os.path.join("./runs/detect/"+experiment_name,'results.csv')
            mlflow.log_artifact(path_model_weights)
            mlflow.log_artifact(path_model_metrics)


        mlflow.end_run()
