from scipy.stats import norm
from joblib import Parallel, delayed
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix
import numpy as np
import json, os
import pandas as pd
from collections import OrderedDict
import scipy.stats as stats

def metric_compare(y_true, y_pred1, y_pred2, n_samples=1000, n_jobs=-1, metric = roc_auc_score):
    def inner():
        y_true_res, y_pred1_res, y_pred2_res = resample(y_true, y_pred1, y_pred2, stratify=y_true)
        # y_true_res, y_pred1_res, y_pred2_res = resample(y_true, y_pred1, y_pred2)
        return metric(y_true_res, y_pred1_res) - metric(y_true_res, y_pred2_res)

    bootstrap_estimates = Parallel(n_jobs=n_jobs)(delayed(inner)() for _ in range(n_samples))
    bootstrap_std = np.std(bootstrap_estimates)
    stastistic = (metric(y_true, y_pred1) - metric(y_true, y_pred2)) / bootstrap_std
    pval = norm.cdf(-stastistic)
    return pval

def metric_compare_v2(y_true, y_pred1, y_pred2, n_samples=100, n_jobs=-1, metric = roc_auc_score):
    '''
    wilcoxon
    '''
    def inner():
        y_true_res, y_pred1_res, y_pred2_res = resample(y_true, y_pred1, y_pred2, stratify=y_true)
        # y_true_res, y_pred1_res, y_pred2_res = resample(y_true, y_pred1, y_pred2)
        return metric(y_true_res, y_pred1_res), metric(y_true_res, y_pred2_res)

    bootstrap_estimates = Parallel(n_jobs=n_jobs)(delayed(inner)() for _ in range(n_samples))
    bootstrap_estimates = np.array(bootstrap_estimates)
    statistics, pval = stats.wilcoxon(bootstrap_estimates[:,0], bootstrap_estimates[:, 1])
    return pval

def confidence_interval(y_true, y_pred, n_samples=1000, n_jobs=-1, metric = roc_auc_score, alpha=100-95):
    def inner():
        y_true_res, y_pred_res = resample(y_true, y_pred, stratify=y_true)
        return metric(y_true_res, y_pred_res)

    bootstrap_metrics = Parallel(n_jobs=n_jobs)(delayed(inner)() for _ in range(n_samples))
    lower_ci = np.percentile(bootstrap_metrics, alpha/2)
    upper_ci = np.percentile(bootstrap_metrics, 100 - alpha/2)
    return lower_ci, upper_ci