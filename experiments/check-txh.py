import pdb
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from logger import logger
from politifactDataloader import Dataset
np.random.seed(103)

data_root='../no-tgt-coref'
coref_ext=''
elasticsearch_dir = os.path.join(data_root, f'elasticsearch{coref_ext}')

dataset = Dataset(data_root, coref_ext)

transcripts_names = list(dataset.transcripts.keys())
transcripts_names.sort()

train_transcripts_names = transcripts_names[:50]
train_transcripts_names.remove('20170512_Trump_NBC_holt_interview')
test_transcripts_names = transcripts_names[50:]

for transcript_name in transcripts_names:
    transcript = dataset.transcripts[transcript_name]
    elasticsearch_transcript = np.load(os.path.join(elasticsearch_dir, transcript_name+'.npz'), allow_pickle=True)

    vclaim_seleced_idxs = elasticsearch_transcript['text'][0][:, :100].astype('int')
    