import argparse
import pdb
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC

# from keras.models import Sequential
# from keras.layers import Dense
import imblearn 

from logger import logger
from politifactDataloader import Dataset


np.random.seed(5)

def load_predictions(fpath):
  scores = []
  with open(fpath) as f:
    for line in (f):
      scores.append(float(line))
  return np.array(scores)


def load_elasticsearch(fpath, index=5):
  scores = []
  with open(fpath) as f:
    for line in (f):
      d = {index: 0}
      for x in line.split(' ')[2:]:
        d[int(x.split(':')[0])] = float(x.split(':')[1])
      scores.append(d[index])
  return np.array(scores)

def load_labels(fpath):
  scores = []
  with open(fpath) as f:
    for line in (f):
      scores.append(int(line.split(' ')[0]))
  return np.array(scores)

def evaluate(labels, predictions):
  accuracy = metrics.accuracy_score(labels, predictions)
  f1 = metrics.f1_score(labels, predictions)
  AUC = metrics.roc_auc_score(labels, predictions)
  recall = metrics.recall_score(labels, predictions)
  precision = metrics.precision_score(labels, predictions)
  score = {
  'f1': f1, 
  'recall': recall, 
  'precision': precision,
  'accuracy': accuracy, 
  'AUC': AUC}
  return score

def get_train_test(args, transcripts_names, elasticsearch=False, test=False, force_positives=False):
  # Load all data
  logger.info("Creating data")
  input = []
  labels = []
  reranker_dir = args.reranker_dir
  if test:
    reranker_dir = 'tmp'
  for transcript_name in transcripts_names:
    if elasticsearch:
      input_ = load_elasticsearch(os.path.join(reranker_dir, transcript_name+'.qid'), index=5)
    else:
      input_ = load_predictions(os.path.join(reranker_dir, transcript_name+'.qid.predict'))
    label_ = load_labels(os.path.join(reranker_dir, transcript_name+'.qid'))
    input_ = input_.reshape((-1, 100))
    label_ = label_.reshape((-1, 100))
    for i in range(len(input_)):
      indices = np.argsort(input_[i])[::-1]
      input_[i] = input_[i][indices]
      label_[i] = label_[i][indices]
    input.append(input_)
    labels.append(label_)
  input = np.concatenate(input, axis=0)
  labels = np.concatenate(labels, axis=0)
  if force_positives:
    input_ = input.copy()
    labels_ = labels.copy()
    for i, (x, l) in enumerate(zip(input, labels)):
      pos = np.where(l)[0]
      inds = np.concatenate([pos, np.delete(np.arange(len(x)), pos)]).squeeze()
      input_[i] = x[inds]
      labels_[i] = l[inds]
    input = input_
    labels = labels_
  return input, labels

def run(args):
  if not os.path.exists(args.data_root):
    logger.error("Data directory (%s) doesnt exist"%args.data_root)
    exit()
  if not os.path.exists(args.reranker_dir):
    logger.error("Reranker embeddings directory (%s) doesnt exist"%args.reranker_dir)
    exit()

  dataset = Dataset(args.data_root)
  
  transcripts_names = list(dataset.transcripts.keys())
  transcripts_names.sort()
  train_transcripts_names = transcripts_names[:50]
  ACL_test_transcripts_names = transcripts_names[50:60]
  COLING_test_transcripts_names = transcripts_names[60:]

  print(ACL_test_transcripts_names)
  print(COLING_test_transcripts_names)
  # COLING_test_transcripts_names = ['20170512_Trump_NBC_holt_interview']

  # for t in COLING_test_transcripts_names:
  #   print(t, dataset.transcripts[t][dataset.transcripts[t].vclaims.map(lambda x: len(x) != 0)])

  mlp = False
  elasticsearch = False

  input, labels = get_train_test(args, COLING_test_transcripts_names, 
    elasticsearch=elasticsearch, force_positives=False)
  pdb.set_trace()
  # X_test, y_test = get_train_test(args, ['20170512_Trump_NBC_holt_interview'], elasticsearch=elasticsearch)
  

  # X_test_man, y_test_man = get_train_test(args, ['man-20170512_Trump_NBC_holt_interview'], 
  #   elasticsearch=elasticsearch, test=True, force_positives=True)
  # X_test_old, y_test_old = get_train_test(args, ['old-20170512_Trump_NBC_holt_interview'], 
  #   elasticsearch=elasticsearch, test=True, force_positives=True)
  
  
  # print(labels)

  # print(input.shape, labels.shape)

  # indices = np.where(labels.sum(axis=1))[0]
  # if args.shuffle:
  #   logger.info("Shuffling data")
  #   np.random.shuffle(indices)
  # train_indices = indices[:int(len(indices) * (1 - 0.2))]
  # test_indices = indices[int(len(indices) * (1 - 0.2)):]

  # indices = np.where(labels.sum(axis=1) == 0)[0]
  # if args.shuffle:
  #   logger.info("Shuffling data")
  #   np.random.shuffle(indices)
  # train_indices = np.concatenate((train_indices, indices[:int(len(indices) * (1 - args.test_ratio))]))
  # test_indices = np.concatenate((test_indices, indices[int(len(indices) * (1 - args.test_ratio)):]))
  # np.random.shuffle(train_indices)
  # np.random.shuffle(test_indices)


  indices = np.arange(len(labels))
  if args.shuffle:
    logger.info("Shuffling data")
    np.random.shuffle(indices)
  train_indices = indices[:int(len(indices) * (1 - 0.2))]
  test_indices = indices[int(len(indices) * (1 - 0.2)):]

  X_train = input[train_indices]
  X_test = input[test_indices]
  y_train = labels[train_indices]
  y_test = labels[test_indices]


  print(X_train.shape, y_train.shape)
  print(y_train.sum(axis=1))
  print(y_train.sum())
  # print(X_test.shape, y_test.shape)
  # print(y_test.sum(axis=1))
  # print(y_test.sum())

  K = [3, 4, 5, 10, 20, 50]
  results = {
    'train': {},
    "test": {},
    "test-old": {},
    "test-man": {},
  }
  results_tsv = ''
  random_tsv = ''
  majority_tsv = ''

  for k in K:

    if mlp:
      model = Sequential()
      model.add(Dense(100, input_dim=k, activation='relu'))
      model.add(Dense(50, activation='relu'))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    else:
      # pipeline = imblearn.pipeline.make_pipeline(imblearn.under_sampling.NearMiss(version=2),
      pipeline = imblearn.pipeline.make_pipeline(imblearn.over_sampling.SVMSMOTE(),
        SVC(kernel='linear'))

    X = X_train[:, :k]
    Y = y_train[:, :k]
    Y = np.where(Y.sum(axis=1), 1, 0)
    Y_ = Y
    print(X.shape, Y.shape)
    if mlp:
      training_generator, steps_per_epoch = imblearn.keras.balanced_batch_generator(
        X, Y, sampler=imblearn.under_sampling.NearMiss(), batch_size=10, random_state=42)
      model.fit_generator(generator=training_generator,
                          steps_per_epoch=steps_per_epoch,
                          epochs=10, verbose=0)
    else:
      pipeline.fit(X, Y)

    if mlp:
      predictions = model.predict_classes(X, verbose=1, batch_size=2048)
    else:
      predictions = pipeline.predict(X)
      print(imblearn.metrics.classification_report_imbalanced(Y, predictions))
    print(predictions.sum())
    
    train_results = evaluate(Y, predictions)
    results['train']['%d'%(k)] = train_results
    # results_tsv += f"train K=({k})\t"
    # results_tsv += f"{Y.shape[0]}\t{Y.sum()}\n"
    # results_tsv += f"{train_results['f1']}\t{train_results['recall']}\t{train_results['precision']}\t{train_results['accuracy']}\t{train_results['AUC']}\n"

    X = X_test[:, :k]
    Y = y_test[:, :k]
    Y = np.where(Y.sum(axis=1), 1, 0)
    if mlp:
      predictions = model.predict_classes(X, verbose=1, batch_size=2048)
    else:
      predictions = pipeline.predict(X)
      print(imblearn.metrics.classification_report_imbalanced(Y, predictions))
    print(predictions.sum())
    test_results = evaluate(Y, predictions)
    results['test']['%d'%(k)] = test_results
    # results_tsv += f"test K=({k})\n"
    results_tsv += f"all\n"
    results_tsv += f"{k}\t{Y.shape[0]}\t{Y.sum()}\t"
    results_tsv += f"{test_results['f1']}\t{test_results['recall']}\t{test_results['precision']}\t{test_results['accuracy']}\t{test_results['AUC']}\n"


    # X = X_test_old[:, :k]
    # Y = y_test_old[:, :k]
    # Y = np.where(Y.sum(axis=1), 1, 0)
    # if mlp:
    #   predictions = model.predict_classes(X, verbose=1, batch_size=2048)
    # else:
    #   predictions = pipeline.predict(X)
    #   print(imblearn.metrics.classification_report_imbalanced(Y, predictions))
    # print(predictions.sum())
    # test_results = evaluate(Y, predictions)
    # results['test-old']['%d'%(k)] = test_results
    # # results_tsv += f"test K=({k})\n"
    # results_tsv += f"old\n"
    # results_tsv += f"{k}\t{Y.shape[0]}\t{Y.sum()}\t"
    # results_tsv += f"{test_results['f1']}\t{test_results['recall']}\t{test_results['precision']}\t{test_results['accuracy']}\t{test_results['AUC']}\n"

    # X = X_test_man[:, :k]
    # Y = y_test_man[:, :k]
    # Y = np.where(Y.sum(axis=1), 1, 0)
    # if mlp:
    #   predictions = model.predict_classes(X, verbose=1, batch_size=2048)
    # else:
    #   predictions = pipeline.predict(X)
    #   print(imblearn.metrics.classification_report_imbalanced(Y, predictions))
    # print(predictions.sum())
    # test_results = evaluate(Y, predictions)
    # results['test-man']['%d'%(k)] = test_results
    # # results_tsv += f"test K=({k})\n"
    # results_tsv += f"man\n"
    # results_tsv += f"{k}\t{Y.shape[0]}\t{Y.sum()}\t"
    # results_tsv += f"{test_results['f1']}\t{test_results['recall']}\t{test_results['precision']}\t{test_results['accuracy']}\t{test_results['AUC']}\n"



    # predictions = np.random.choice(2, len(X))
    # random_results = evaluate(Y, predictions)
    # random_tsv += f"random\t{k}\t{Y.shape[0]}\t{Y.sum()}\t"
    # random_tsv += f"{random_results['f1']}\t{random_results['recall']}\t{random_results['precision']}\t{random_results['accuracy']}\t{random_results['AUC']}\n"

    # predictions = np.random.choice(2, len(X), p=[(len(Y_) - Y_.sum())/len(Y_), Y_.sum()/len(Y_)])
    # majority_results = evaluate(Y, predictions)
    # majority_tsv += f"majority\t{k}\t{Y.shape[0]}\t{Y.sum()}\t"
    # majority_tsv += f"{majority_results['f1']}\t{majority_results['recall']}\t{majority_results['precision']}\t{majority_results['accuracy']}\t{majority_results['AUC']}\n"

  logger.info("Dumping results to (%s) file"%args.out_file)
  with open(args.out_file, 'w') as f:
    json.dump(results, f, indent=4)
  with open(args.out_file+'.tsv', 'w') as f:
    f.write(results_tsv)
    # f.write(random_tsv)
    # f.write(majority_tsv)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get elasticsearch scores')
  parser.add_argument('--data-root', '-d', required=True, 
    help='Path to the dataset directory.')
  parser.add_argument('--reranker-dir', '-r', required=True, 
    help='Path to the reranker data directory')
  parser.add_argument('--out-file', '-o', required=True, 
    help='Name of output file were results will be dumped as a json file')
  parser.add_argument('--shuffle', default=False, action='store_true',
    help='Flag set if you want the data to be shuffled')
  parser.add_argument('--test-ratio', default=.2, type=float,
    help='Ratio of sentences that will be used for testing')

  args = parser.parse_args()
  run(args)
