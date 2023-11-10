#!/usr/bin/env python3
"""
logger
"""

from collections import OrderedDict
import numpy as np

def logger_all(metric, n_tasks=None):
    log_metric = OrderedDict()
    log_metric[metric] = np.zeros([n_tasks, n_tasks])
    log_metric['final_acc'] = 0.
    log_metric['final_forget'] = 0.
    log_metric['final_la'] = 0.
    return log_metric

def logger_eval(metric):
    log_metric = OrderedDict()
    log_metric[metric] = []
    return log_metric
    
def per_task_summary(log_metric, metric, task_id=0, task_t=0, value=0):
    if metric is 'acc':
        log_metric[metric][task_t, task_id] = value
    else:
        log_metric[metric] = value


