from glob import glob
from os import path
import torch
import numpy as np

exp_path = '/home//drone_landing/executions'

summary_search = path.join(exp_path, '*', 'test_results', 'metrics_summary')

summary_paths = glob(summary_search)

summaries = []
exp_names = []
for p in summary_paths:
    summary = torch.load(p, map_location='cpu')
    summaries.append(summary)
    exp_names.append(p.split(path.sep)[-1])

acc = [r['acc'] for r in summaries]
jcc = [r['jcc'] for r in summaries]
pre = [r['pre'] for r in summaries]
f1 = [r['f1'] for r in summaries]
conf = [r['confusion'] for r in summaries]

acc = np.asarray(acc)
jcc = np.vstack(jcc)
pre = np.vstack(pre)
f1 = np.vstack(f1)

jcc_exp_ids, jcc_counts = np.unique(np.argmax(jcc, axis=0), return_counts=True)
jcc_winner = jcc_exp_ids[np.argmax(jcc_counts)]
pre_exp_ids, pre_counts = np.unique(np.argmax(pre, axis=0), return_counts=True)
pre_winner = pre_exp_ids[np.argmax(pre_counts)]
f1_exp_ids, f1_counts = np.unique(np.argmax(f1, axis=0), return_counts=True)
f1_winner = f1_exp_ids[np.argmax(f1_counts)]

