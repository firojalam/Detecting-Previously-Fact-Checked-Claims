import pdb
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import datasets

from logger import logger
from politifactDataloader import Dataset
np.random.seed(103)

def dump_svmlight(inputs, labels, out_fpath, limit=10):
  X = inputs[:, :limit, :]
  y = labels[:, :limit]
  qid = np.mgrid[0:len(X), 0:limit][0]

  X = X.reshape((-1, X.shape[-1]))
  y = y.reshape((-1))
  qid = qid.reshape((-1))

  datasets.dump_svmlight_file(X, y, out_fpath, zero_based=False, query_id=qid)

def load_predictions(fpath):
  scores = []
  with open(fpath) as f:
    for line in tqdm(f):
      scores.append(float(line.split()))
  return np.arary(scores)

# HARD CODED!
def create_train_test_svmlight(dataset, out_dir, reranker_dir, 
                               train_transcripts_names, test_transcripts_names, 
                               before=0, after=0):
  X_train, y_train = [], []
  X_test, y_test = [], []

  for transcript_name in train_transcripts_names:
    transcript = dataset.transcripts[transcript_name]
    data = np.load(os.path.join(reranker_dir, transcript_name+'.npz'))
    all_inputs = data['input']
    all_labels = data['labels']
    indices = np.where(transcript.vclaims.map(lambda x: len(x) != 0))[0] - before
    inputs = []
    # labels = []
    all_indices = []
    for i in range(0, before+after+1, 1):
      indices_ = indices + i
      indices_ = np.where(indices_ < 0, 0, indices_)
      indices_ = np.where(indices_ > (len(transcript) - 1), (len(transcript) - 1), indices_)
      inputs.append(all_inputs[indices_])
      # labels.append(all_labels[indices_])
      all_indices.append(indices_.reshape((-1, 1)))
    inputs = np.concatenate(inputs, axis=2)
    labels = all_labels[np.where(transcript.vclaims.map(lambda x: len(x) != 0))[0]]
    all_indices = np.concatenate(all_indices, axis=1)
    X_train.append(inputs)
    y_train.append(labels)
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)
  print(X_train.shape)
  print(y_train.shape)
  for transcript_name in test_transcripts_names:
    transcript = dataset.transcripts[transcript_name]
    data = np.load(os.path.join(reranker_dir, transcript_name+'.npz'))
    all_inputs = data['input']
    all_labels = data['labels']
    indices = np.where(transcript.vclaims.map(lambda x: len(x) != 0))[0] - before
    inputs = []
    # labels = []
    all_indices = []
    for i in range(0, before+after+1, 1):
      indices_ = indices + i
      indices_ = np.where(indices_ < 0, 0, indices_)
      indices_ = np.where(indices_ > (len(transcript) - 1), (len(transcript) - 1), indices_)
      inputs.append(all_inputs[indices_])
      # labels.append(all_labels[indices_])
      all_indices.append(indices_.reshape((-1, 1)))
    inputs = np.concatenate(inputs, axis=2)
    labels = all_labels[np.where(transcript.vclaims.map(lambda x: len(x) != 0))[0]]
    all_indices = np.concatenate(all_indices, axis=1)
    X_test.append(inputs)
    y_test.append(labels)
  X_test = np.concatenate(X_test, axis=0)
  y_test = np.concatenate(y_test, axis=0)
  print(X_test.shape)
  print(y_test.shape)
  dump_svmlight(X_train, y_train, os.path.join(out_dir, f'train.qid'), limit=100)
  dump_svmlight(X_test, y_test, os.path.join(out_dir, f'test.qid'), limit=100)

def create_transcript_svmlight(dataset, out_dir, reranker_dir, transcripts_names):

  for transcript_name in transcripts_names:
    transcript = dataset.transcripts[transcript_name]
    data = np.load(os.path.join(reranker_dir, transcript_name+'.npz'))
    inputs = data['input']
    labels = data['labels']
    dump_svmlight(inputs, labels, os.path.join(out_dir, transcript_name+'.qid'), limit=100)


def run(args):
  if not os.path.exists(args.data_root):
    logger.error("Data directory (%s) doesnt exist"%args.data_root)
    exit()
  if not os.path.exists(args.reranker_dir):
    logger.error("Reranker embedding directory (%s) doesnt exist"%args.reranker_dir)
    exit()
  
  if not os.path.exists(args.out_dir):
    logger.warning("Output directory (%s) doesnt exist"%args.out_dir)
    os.makedirs(args.out_dir)

  dataset = Dataset(args.data_root, args.coref_ext, concat_before=args.concat_before, 
    concat_after=args.concat_after)
  
  transcripts_names = list(dataset.transcripts.keys())
  transcripts_names.sort()
  train_transcripts_names = transcripts_names[:50]
  train_transcripts_names.remove('20170512_Trump_NBC_holt_interview')
  ACL_test_transcripts_names = transcripts_names[50:]
  # COLING_test_transcripts_names = transcripts_names[60:]
  # COLING_test_transcripts_names = ['20170512_Trump_NBC_holt_interview']

  create_train_test_svmlight(dataset, args.out_dir, args.reranker_dir, 
                             train_transcripts_names, ACL_test_transcripts_names, 
                             before=args.before, after=args.after)

  # create_transcript_svmlight(dataset, args.out_dir, args.reranker_dir, COLING_test_transcripts_names)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get elasticsearch scores')
  parser.add_argument('--data-root', '-d', required=True, 
    help='Path to the dataset directory.')
  parser.add_argument('--out-dir', '-o', required=True, 
    help='Path to the output directory')
  parser.add_argument('--reranker-dir', '-r', required=True, 
    help='Path to the reranker embeddings directory')
  parser.add_argument('--before', default=0, type=int,
    help='Number of sentences you take as context from BEFRORE the sentence')
  parser.add_argument('--after', default=0, type=int,
    help='Number of sentences you take as context from AFTER the sentence')
  parser.add_argument('--concat-before', default=0, type=int,
    help='Number of sentences concatenated before the input sentence in a transcript')
  parser.add_argument('--concat-after', default=0, type=int,
    help='Number of sentences concatenated after the input sentence in a transcript')
  parser.add_argument('--coref-ext', default='', type=str,
    help='If using co-referenced resolved transcripts it gives the resolved coreference data. ({data}/transcripts{EXT} provide EXT)')

  args = parser.parse_args()
  run(args)