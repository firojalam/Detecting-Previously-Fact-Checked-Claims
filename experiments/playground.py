from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os
import glob
import pdb
import sys
import json

VCLAIMS_COLUMNS = ["vclaim_id", "vclaim", "date", "article_url", "truth_meter", 
"speaker", "title", "text"]
verified_claims_og = pd.read_csv('../no-tgt-coref/verified-claims.tsv', sep='\t', header=None, 
                          names=VCLAIMS_COLUMNS, doublequote=False, 
                          quoting=3)
verified_claims_og.fillna('', inplace=True)

verified_claims = pd.read_csv('../e2e-coref.nw/verified-claims.tsv', sep='\t', header=None, 
                          names=VCLAIMS_COLUMNS, doublequote=False, 
                          quoting=3)
verified_claims.fillna('', inplace=True)

with open('../e2e-coref.nw.text/verified-claims.tsv', 'w') as ff:
	for i, row in tqdm(list(verified_claims.iterrows())):
		r = ''
		
		text = row.text
		vclaim = verified_claims_og.iloc[i].vclaim

		r += f'{row["vclaim_id"]}\t{vclaim}\t{row["date"]}\t{row["article_url"]}\t{row["truth_meter"]}\t{row["speaker"]}\t{row["title"]}\t{text}\n'
		ff.write(r)