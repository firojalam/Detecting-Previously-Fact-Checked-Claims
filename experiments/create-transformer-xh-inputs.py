import sys
import glob 
import numpy as np
import os
import pdb
import json
from politifactDataloader import Dataset
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import nltk 
np.random.seed(103)

data_dir = '../no-tgt-coref'
model='.neuralcoref.all'
out_dir = 'transformer-xh-src-neural'
if not os.path.exists(out_dir):
  os.makedirs(out_dir)


SBERT_dir = f'{data_dir}/SBERT.large.embeddings{model}'
dataset = Dataset(data_dir, transcripts_dir_ext=model)
verifiedClaims = dataset.verified_claims

sbert_vclaims_text = np.load(os.path.join(data_dir, 'vclaim.text.npy'), allow_pickle=True)


all = []

transcripts_names = list(dataset.transcripts.keys())
transcripts_names.sort()

train_transcripts_names = transcripts_names[:50]
train_transcripts_names.remove('20170512_Trump_NBC_holt_interview')
test_transcripts_names = transcripts_names[50:]

for transcript_name in transcripts_names[int(sys.argv[1]):int(sys.argv[2])]:
  print(transcript_name)
  with open(f'{out_dir}/{transcript_name}.json', 'w') as f:
    transcript = dataset.transcripts[transcript_name]
    
    es_path = os.path.join(f'{data_dir}/elasticsearch{model}', transcript_name+'.npz')
    elasticsearch_transcript = np.load(es_path, allow_pickle=True)
    
    sbert_transcript = np.load(os.path.join(SBERT_dir, transcript_name+'.npy'), allow_pickle=True)

    claims_idxs_ = np.arange(len(transcript))[transcript.vclaims.map(lambda x: len(x) != 0)]
    claims_idxs = []
    for claim_idx in claims_idxs_:
      claims_idxs.extend(range(claim_idx-3, claim_idx+2))
    claims_idxs = list(set(claims_idxs))
    claims_idxs.sort()
    # print(claims_idxs_)
    # print(claims_idxs)
    for claim_idx in tqdm(claims_idxs):
      input_claim = transcript.iloc[claim_idx].sentence
      claim_embedding = sbert_transcript[claim_idx].reshape(1, -1)
      for rank, vclaim_id in enumerate(elasticsearch_transcript['text'][0][claim_idx]):
        vclaim_id = int(vclaim_id)
        vclaim_obj = verifiedClaims.loc[verifiedClaims.vclaim_id == vclaim_id].iloc[0]
        label = False      
        if vclaim_id in transcript.iloc[claim_idx].vclaims:
          label = True
        
        article_text = np.array(nltk.sent_tokenize(vclaim_obj.text))
        vclaim_text_embeddings = sbert_vclaims_text[vclaim_id]
        if not len(vclaim_text_embeddings):
          print('weird', vclaim_id)
          selected_text = ['', '', '', '']
        else:
          vclaim_text_score = metrics.pairwise.cosine_similarity(vclaim_text_embeddings, claim_embedding).squeeze()
          vclaim_text_rank = np.argsort(vclaim_text_score)
          n = min(4, len(vclaim_text_embeddings))
          selected_text = list(article_text[vclaim_text_rank][-n:])

        entity  = {
          'input': input_claim, 
          'line_number': int(claim_idx+1),
          'label': label, 
          'rank': rank,
          'transcript_name': transcript_name,
          'vclaim': vclaim_obj.vclaim, 
          'vclaim_id': vclaim_id,
          'vclaim_title': vclaim_obj.title, 
          'vclaim_date': vclaim_obj.date, 
          'vclaim_speaker': vclaim_obj.speaker, 
          'vclaim_truth': vclaim_obj.truth_meter, 
          'vclaim_text': selected_text
        }
        entity_str = json.dumps(entity)
        f.write(entity_str.strip()+'\n')
