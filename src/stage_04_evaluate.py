import argparse
import os
import shutil
import logging
from src.utils.common import *
import joblib
import numpy as np
import sklearn.metrics as metrics
import math

STAGE = "Evaluation"

def evaluate(config_path : str, params_path : str):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]

    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])

    featurized_test_data_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_DATA_TEST"])
      
    model_dir = artifacts["MODEL_DIR"]
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], model_dir)
    model_name = artifacts["MODEL_NAME"]
    model_path = os.path.join(model_dir_path, model_name)

    model = joblib.load(model_path)
    matrix = joblib.load(featurized_test_data_path)

    labels = np.squeeze(matrix[:, 1].toarray())
    X = matrix[:, 2:]

    predictions_probabilities = model.predict_proba(X)
    pred = predictions_probabilities[:, 1]
    
    logging.info(f"labels, predictions: {list(zip(labels, pred))}")

    scores_path = os.path.join(os.getcwd(),config["metrics"]["SCORES"])
    prc_path  = os.path.join(os.getcwd(),config["plots"]["PRC"])
    roc_path = os.path.join(os.getcwd(),config["plots"]["ROC"])


    
    avg_prec = metrics.average_precision_score(labels, pred)
    roc_auc = metrics.roc_auc_score(labels, pred)

    logging.info(f"len of labels: {len(labels)} and predictions: {len(pred)}")
    scores = {
        "avg_prec": avg_prec,
        "roc_auc":roc_auc
    }

    save_json(scores_path, scores)

    precision, recall, prc_threshold = metrics.precision_recall_curve(labels,pred)

    nth_points = math.ceil(len(prc_threshold)//1000)
    prc_list = list(zip(precision,recall,prc_threshold))[::nth_points]
    
    prc_json_points = {
        "prc": [{'precision':p,"recall":r, "threshold":p_t} for p,r,p_t in prc_list]
    }


    save_json(prc_path,prc_json_points)

    fpr, tpr, roc_threshold = metrics.roc_curve(labels, pred)
    roc_points = zip(fpr, tpr, roc_threshold)
    roc_data = {
        "roc": [
            {"fpr": fp, "tpr": tp, "threshold": t}
            for fp, tp, t in roc_points
        ]
    }
    logging.info(f"no. of roc points: {len(list(roc_points))}")
    # logging.info(f"fpr: {fpr}, \ntpr: {tpr}, \nroc_threshold: {roc_threshold}")

    save_json(roc_path, roc_data)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        evaluate(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e