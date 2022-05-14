from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, classification_report, roc_auc_score
from collections import OrderedDict
from sklearn import metrics
import hashlib
from datetime import datetime
import json, os
from batchgenerators.utilities.file_and_folder_operations import save_json

# def z_score_normalization(X, mean=None, std=None):
#     if mean is None or std is None:
#         mean = np.mean(X, axis=0, keepdims=True)
#         std = np.std(X, axis=0, keepdims=True)
#     return (X - mean) / std, mean, std

def aggregate_classification_scores(test_ref_pairs,
                                     identifiers,
                                     nanmean=True,
                                     json_output_file=None,
                                     json_name="",
                                     json_description="",
                                     json_author="Fabian",
                                     json_task="",
                                     num_threads=2,
                                     **metric_kwargs):
    all_scores = OrderedDict()
    for i, (prob, pred, label) in enumerate(test_ref_pairs):
        identifier = identifiers[i]
        case_score = OrderedDict()
        case_score['probability'] = prob
        case_score['pred'] = float(pred) if pred is not None else pred
        case_score['label'] = float(label)
        all_scores[identifier] = case_score

    overall_score = OrderedDict()
    probs = np.array([test_ref_pair[0] for test_ref_pair in test_ref_pairs if test_ref_pair[1] is not None])
    preds = np.array([test_ref_pair[1] for test_ref_pair in test_ref_pairs if test_ref_pair[1] is not None])
    labels = np.array([test_ref_pair[2] for test_ref_pair in test_ref_pairs if test_ref_pair[1] is not None])
    overall_score['accuracy'] = float(metrics.accuracy_score(y_pred=preds, y_true=labels))
    overall_score['recall'] = float(metrics.recall_score(y_pred=preds, y_true=labels))
    overall_score['classification report'] = metrics.classification_report(y_pred=preds, y_true=labels, digits=3)
    overall_score['balanced accuracy'] = float(metrics.balanced_accuracy_score(y_pred=preds, y_true=labels))
    overall_score['precision'] = float(metrics.precision_score(y_pred=preds, y_true=labels))
    overall_score['auc'] =float( metrics.roc_auc_score(y_true=labels, y_score=probs))
    print(overall_score['classification report'])
    all_scores['overall'] = overall_score
    if json_output_file is not None:
        json_dict = OrderedDict()
        json_dict["name"] = json_name
        json_dict["selected variables"] = json_description
        timestamp = datetime.today()
        json_dict["timestamp"] = str(timestamp)
        json_dict["task"] = json_task
        json_dict["author"] = json_author
        json_dict["results"] = all_scores
        json_dict["id"] = hashlib.md5(json.dumps(json_dict).encode("utf-8")).hexdigest()[:12]
        save_json(json_dict, json_output_file)


def LASSO_regression(X, y):
    reg = LassoCV(cv=10, random_state=0).fit(X,y)
    return reg

def LASSO_and_logistic_regression(X_train, y_train, X_test,y_test, colNames, fold, has_LASSO=True):
    # z score normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = colNames
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test)
    X_test.columns = colNames
    # X_train, mean, std = z_score_normalization(X_train)
    # X_test = z_score_normalization(X_test, mean, std)
    index = None
    if has_LASSO:
        # LASSO
        lasso_reg = LASSO_regression(X_train, y_train)
        # y_pred = lasso_reg.predict(X_test)
        coef = pd.Series(lasso_reg.coef_, index=colNames)

        print('Lasso picked ' + str(sum(coef !=0)) + ' variables and eliminated the other ' + str(sum(coef == 0)))

        index = coef[coef != 0].index
        print('fold: ' + str(fold) + ' select ', index.values.tolist())
        X_train = X_train[index]
        X_test = X_test[index]

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_test_pred = classifier.predict(X_test)
    y_test_prob = classifier.predict_proba(X_test)

    signature_train = np.dot(X_train, classifier.coef_.T) + classifier.intercept_
    signature_test = np.dot(X_test, classifier.coef_.T) + classifier.intercept_

    # metrics_dict["accuracy"].append(accuracy_score(y_pred=y_test_pred, y_true=y_test))
    # metrics_dict["balanced accuracy"].append(balanced_accuracy_score(y_pred=y_test_pred, y_true=y_test))
    # metrics_dict["precision"].append(precision_score(y_pred=y_test_pred, y_true=y_test))
    # metrics_dict["recall"].append(recall_score(y_pred=y_test_pred, y_true=y_test))
    # metrics_dict["specificity"].append(classification_report(y_pred=y_test_pred, y_true=y_test, output_dict=True)['0']['recall'])
    # metrics_dict["AUC"].append(roc_auc_score(y_score=y_test_prob[:, 1], y_true=y_test))

    return list(zip(y_test_prob[:, 1], y_test_pred, y_test)), index.values.tolist() if index is not None else None, signature_train, signature_test