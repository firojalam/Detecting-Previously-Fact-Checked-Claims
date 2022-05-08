import os
import glob
import json
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import nltk

from logger import logger

# A function that retures the paths of all the files in the directory
# 	(dir) witht he extension (extension) excluding the file (exclude).


def get_files(dir, exclude='', extension='csv'):
  logger.info("Getting all files from (%s) with the extension (%s) excluding (%s)" % (
    dir, extension, exclude))

  return glob.glob(os.path.join(dir, '[!_%s]*.%s' % (exclude, extension)))


def compare_lists(l1, l2):
  # NOT IMPLIMENTED
  return 0

def load_config(config):
  with open(config, "r") as config_file:
    return json.load(config_file)




def compute_MAP(labels, scores, top_k=-1):

  if len(labels) != len(scores):
    logger.error(
      "Failed computing MAP because leght of labels (%d) and scores (%d) are different"%(len(labels), len(scores)))
    return -1

  if top_k < 0 or top_k > labels.shape[1]:
    top_k = labels.shape[1]

  average_precision_scores = []
  for i, (label, score) in enumerate(zip(labels, scores)):

    sorted_indices = score.argsort()[::-1]
    score = score[sorted_indices]
    label = label[sorted_indices]
    score[top_k:] = 0

    if sum(label) == 0:
      logger.error("Found something when computing MAP no labels (%d)"%sum(label))
      average_precision_score = 0
    else:
      average_precision_score = metrics.average_precision_score(label, score)

    average_precision_scores.append(average_precision_score)

  return np.mean(average_precision_scores), average_precision_scores


def compute_recall(labels, scores, top_k=1):

  if len(labels) != len(scores):
    logger.error(
      "Failed computing RECALL because leght of labels and scores are different")
    return -1

  if top_k < 0 or top_k > labels.shape[1]:
    top_k = labels.shape[1]

  recalls = []
  for i, (label, score) in enumerate(zip(labels, scores)):
    sorted_indices = score.argsort()[::-1][:top_k]

    if sum(label) == 0:
      logger.error("Found something when computing RECALL no labels (%d)"%sum(label))
      recall = 0
    else:
      max_score = np.sum(label)
      recall = label[sorted_indices].sum()/max_score

    recalls.append(recall)
  return np.mean(recalls), recalls


def compute_precision(labels, scores, top_k=1):

  if len(labels) != len(scores):
    logger.error(
      "Failed computing PRECISION because leght of labels and scores are different")
    return -1

  if top_k < 0 or top_k > labels.shape[1]:
    top_k = labels.shape[1]

  precisions = []
  for i, (label, score) in enumerate(zip(labels, scores)):
    sorted_indices = score.argsort()[::-1][:top_k]
    precision = label[sorted_indices].sum()/top_k
    precisions.append(precision)

  return np.mean(precisions), precisions

def compute_MRR(labels, scores, top_k=1):

  if len(labels) != len(scores):
    logger.error(
      "Failed computing MRR because leght of labels and scores are different")
    return -1

  MRRs = []
  for i, (label, score) in enumerate(zip(labels, scores)):

    sorted_indices = score.argsort()[::-1]
    sorted_label = label[sorted_indices]
    true_positions = np.where(sorted_label)[0]

    if len(true_positions) == 0:
      logger.error("Found something when computing MRR no labels (%d)"%sum(label))
      MRR = 0
    else:
      MRR = 1.0/(true_positions[0] + 1)


    MRRs.append(MRR)
  return np.mean(MRRs), MRRs


def compute_top_positives(labels, scores, top_k=1):

  if len(labels) != len(scores):
    logger.error(
      "Failed computing top_positive because leght of labels and scores are different")
    return -1

  if top_k < 0 or top_k > labels.shape[1]:
    top_k = labels.shape[1]

  top_positives = []
  for i, (label, score) in enumerate(zip(labels, scores)):

    sorted_indices = score.argsort()[::-1][:top_k]

    if sum(label) == 0:
      logger.error("Found something when computing top_positive no labels (%d)"%sum(label))
      top_positive = 0
    else:
      top_positive = 1 if label[sorted_indices].sum() else 0

    top_positives.append(top_positive)
  return np.mean(top_positives), top_positives
