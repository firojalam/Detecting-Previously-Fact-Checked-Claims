import os
import sys
import pdb
import pandas as pd
import numpy as np
from tqdm import tqdm 

from politifactDataloader import Dataset
from logger import logger
import utils

data_dir = sys.argv[1]
es_dir = os.path.join(data_dir, 'elasticsearch')

def load_ranksvm_predictions(fpath):
  scores = []
  with open(fpath) as f:
    for line in tqdm(f):
      scores.append(float(line.strip()))
  return np.array(scores)

def load_elasticsearch_predictions(fpath, type='text'):
  scores = []
  with open(fpath) as f:
    for line in tqdm(f):
      scores.append(float(line.strip()))
  return np.array(scores)


def get_labels_predictions(scores, transcripts_names, dataset, type=''):
  verified_claims = dataset.verified_claims

  labels = np.zeros((len(scores), len(dataset.verified_claims)))
  predictions = np.zeros((len(scores), len(dataset.verified_claims)))

  i = 0
  for transcript_name in transcripts_names:
    transcript = dataset.transcripts[transcript_name]

    claims_idxs = np.arange(len(transcript))[transcript.vclaims.map(lambda x: len(x) != 0)]
    transcript_labels = dataset.get_labels_(transcript, claims_idxs, type=type)
    # print(claims_idxs)
    es_path = os.path.join(es_dir+f'{"." if model else ""}{model}', transcript_name+'.npz')
    # print(es_path)
    elasticsearch_transcript = np.load(es_path, allow_pickle=True)
    vclaim_seleced_idxs = elasticsearch_transcript['text'][0][:, :100].astype('int')
    # pdb.set_trace()
    for k, claims_idx in enumerate(claims_idxs):
      labels[i] = transcript_labels[k]
      for j, vclaim_idx in enumerate(vclaim_seleced_idxs[claims_idx]):
        predictions[i][vclaim_idx] = scores[i][j]
      i += 1

  non_empty = np.where(labels.sum(axis=1))[0]
  labels = labels[non_empty]
  predictions = predictions[non_empty]

  return labels, predictions

metrics = {
  "MAP": -1,
  "MRR": -1, 
  # "top_positives": [1, 2, 3, 5, 10, 20, 50, -1],
  # "recall": [1, 2, 3, 5, 10, 20, 50, -1], 
  # "precision": [1, 2, 3, 5, 10, 20, 50, -1]
}

model = sys.argv[2]
dataset = Dataset(f"{data_dir}",  f'{"." if model else ""}{model}')
dir_ = f'{data_dir}/bug-bug-ranksvm-txh{"." if model else ""}{model}'

results = f'{dir_}\n'

for metric in metrics:
  results += f'{metric}\n'

  for b_a in ['0.0', '1.1', '3.1']:
    dir = f'{dir_}.{b_a}'
    print(dir)
    results += f'{b_a}\t'
    
    if not os.path.exists(dir):
      logger.error(f'No directory found ({dir})')
      results += '\t\t\n'
      continue
    
    try:
      test_scores = load_ranksvm_predictions(f'{dir}/test.qid.predict').reshape((-1, 100))
      # print(test_scores)
    except:
      logger.error(f'Cannot load results from ({dir})')
      results += '\t\t\n'
      continue


    transcripts_names = list(dataset.transcripts.keys())
    transcripts_names.sort()
    train_transcripts_names = transcripts_names[:50]
    train_transcripts_names.remove('20170512_Trump_NBC_holt_interview')
    test_transcripts_names = transcripts_names[50:]

    for dtype in ['test']:
      for type in ['', 'CLEAN', 'CLEAN_HARD', 'PART_OF', 'CONTEXT_DEPENDENT']:
        logger.info(f'Getting labels and predictions for {dtype} and type {type}')
        try:
          labels, predictions = get_labels_predictions(eval(f'{dtype}_scores'), 
              eval(f'{dtype}_transcripts_names'), dataset, type=type)
          # results += f'{dtype}\t{type}\t{len(labels)}\t'
          # results += f'{dtype}\t{type}\t{len(labels)}\t'
          logger.info(f'Getting various scores for {dtype} and type {type}')
          k = metrics[metric]
          score, _ = eval(f'utils.compute_{metric}')(labels, predictions, top_k=k)
        except:
          logger.error(f"COULDNT COMPUTE THE SCORE {b_a} {type}")
          score = 0
        results += f'{score}\t'
    results += '\n'
  # print(results)
with open(f'../results/all-results', 'a') as f:
  f.write(results)


