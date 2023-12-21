from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def update_model(model : Pipeline) -> None:
    dump(model, 'model/model.pkl')


def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
    with open("report.txt", "w") as report_file:
        report_file.write("# Reporte del pipeline del modelo "+"\n")

        for key, value in model.named_steps.items():
            report_file.write(f"### {key}:{value.__repr__()}"+"\n")

        report_file.write(f'### Train Score: {train_score}'+'\n')
        report_file.write(f'### Test Score: {test_score}'+'\n')
        report_file.write(f'### Validation Score: {validation_score}'+'\n')

def get_model_performance_test_set(y_real: pd.Series, Y_pred: pd.Series) -> None:
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=Y_pred, y=y_real, ax=ax)
    ax.set_xlabel("Predic")
    ax.set_ylabel("Real")
    fig.savefig("prediction_behavior.png")
