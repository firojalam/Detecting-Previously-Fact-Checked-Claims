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


data_dir = '../e2e-coref.nw'
dataset = Dataset(data_dir, '.neuralcoref.all')
verifiedClaims = dataset.verified_claims

out_dir = '../e2e-coref.nw/bug-transformer-XH.neuralcoref.all'

if not os.path.exists(out_dir):
	# logger.warning("Output directory (%s) doesnt exist"%out_dir)
	os.makedirs(out_dir)


all = []

transcripts_names = list(dataset.transcripts.keys())
transcripts_names.sort()

train_transcripts_names = transcripts_names[:50]
train_transcripts_names.remove('20170512_Trump_NBC_holt_interview')
test_transcripts_names = transcripts_names[50:]


for ffiles in glob.glob('transformer-xh-src-neural-tgt-e2enw-bug_graph_output_labels/*.jsonl'):
	# print(ffiles)
	
	transcript_name = os.path.basename(ffiles)[:-len('_classified.jsonl')]

	data = {}
	sflies = os.path.join('transformer-xh-src-neural-tgt-e2enw-bug', f'{transcript_name}.json')
	print(sflies)
	print(ffiles)
	with open(ffiles, 'r') as ff:
		with open(sflies, 'r') as sf:
			for (sl, fl) in zip(sf.readlines(), ff.readlines()):
				sd = json.loads(sl)
				fd = json.loads(fl)
				if sd['input'] not in data:
					data[sd['input']] = {
						'vclaim_ids': [-1] * 100, 
						'scores':[(0, 0, 0)] * 100
					}
				data[sd['input']]['vclaim_ids'][sd['rank']] = sd['vclaim_id']
				data[sd['input']]['scores'][sd['rank']] = (fd['not_enough_info'], 
														   fd['supports'], 
														   fd['refutes'])


# for transcript_name in tqdm(train_transcripts_names):
	transcript = dataset.transcripts[transcript_name]
	out = ''
	# es_path = os.path.join(f'{data_dir}/elasticsearch.scores.100', transcript_name+'.npz')
	# elasticsearch_transcript = np.load(es_path, allow_pickle=True)
	
	claims_idxs_ = np.arange(len(transcript))[transcript.vclaims.map(lambda x: len(x) != 0)]
	claims_idxs = []
	for claim_idx in claims_idxs_:
		claims_idxs.extend(range(claim_idx-3, claim_idx+2))
	claims_idxs = list(set(claims_idxs))
	# claims_idxs = claims_idxs_
	claims_idxs.sort()
	for claim_idx in claims_idxs:
		input_claim = transcript.iloc[claim_idx].sentence
		entry = data[input_claim]
		for i in range(100):
			vclaim_id = entry['vclaim_ids'][i]
			scores = entry['scores'][i]
			out += f"{claim_idx+1}\t{vclaim_id}\t{scores[0]}\t{scores[1]}\t{scores[2]}\n"

	out_path = os.path.join(out_dir, transcript_name+'.tsv')
	with open(out_path, 'w') as outf:
		outf.write(out)
