import os
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]  # root directory absolute path


def save_model(experiment_name: str):
    """ saves the weights of trained model to the models directory """
    if os.path.isdir('runs'):
        model_weights = experiment_name + "/weights/best.pt"
        path_model_weights = os.path.join( "runs/detect", model_weights)

        shutil.copyfile(src=path_model_weights, dst=f'{ROOT_DIR}/models/best.pt')


def save_metrics_and_params(experiment_name: str) -> None:
    """ saves training metrics, params and confusion matrix to the reports directory """
    if os.path.isdir('runs'):
        path_metrics = os.path.join( "runs/detect", experiment_name)

        # save experiment training metrics
        shutil.copyfile(src=f'{path_metrics}/results.csv', dst='./reports/results.csv')

        # save the confusion matrix associated to the training experiment
        shutil.copyfile(src=f'{path_metrics}/confusion_matrix.png', dst='./reports/confusion_matrix.png')

        # save training params
        shutil.copyfile(src=f'{path_metrics}/args.yaml', dst='./reports/args.yaml')
