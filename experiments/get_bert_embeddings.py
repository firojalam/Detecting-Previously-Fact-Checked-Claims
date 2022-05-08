import argparse
import pdb
import json
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import nltk 

import torch 
import transformers 
from sentence_transformers import SentenceTransformer

import utils
from logger import logger
from BERT import main as BERT
from politifactDataloader import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode(sentences, model, tokenizer, config, outfile, show_progress_bar=False):
  if tokenizer:
    sentences_embeddings = BERT.encode_transformers(sentences, model, tokenizer, 
                                                    config['pretrained_weights'], 
                                                    out_file=config[outfile], 
                                                    pooling_layer=config['pooling_layer'], 
                                                    pooling_strategy=config['pooling_strategy'])
  else:
    sentences_embeddings = BERT.encode_sentence_bert(sentences, model, config[outfile], 
      show_progress_bar=show_progress_bar)
  return sentences_embeddings


def run(args):
  if not os.path.exists(args.data_root):
    logger.error("Data directory (%s) doesnt exist"%args.data_root)
    exit()
  if not os.path.exists(args.config):
    logger.error("Config file (%s) doesnt exist"%args.config)
    exit()
  if not os.path.exists(args.out_dir):
    logger.warning("Output directory (%s) doesnt exist"%args.out_dir)
    os.makedirs(args.out_dir)
  dataset = Dataset(args.data_root, args.coref_ext, concat_before=args.concat_before, 
    concat_after=args.concat_after)
  config = utils.load_config(args.config)

  model, tokenizer = BERT.get_model(config['pretrained_weights'])
  verifiedClaims = dataset.verified_claims
  verifiedClaims = verifiedClaims.replace(np.nan, '', regex=True)

  if 'vclaim' in args.input:
    logger.info('Getting vclaim statement embeddings')
    article_embeddings = encode(verifiedClaims.vclaim, model, tokenizer, config, 'temp', 
      show_progress_bar=True)
    np.save(os.path.join(args.data_root, 'vclaim.npy'), article_embeddings)
  
  if 'title' in args.input:
    print(verifiedClaims[verifiedClaims.title.map(lambda x: type(x) != str)])

    logger.info('Getting article vclaim title embeddings')
    article_embeddings = encode(verifiedClaims.title, model, tokenizer, config, 'temp', 
      show_progress_bar=True)
    np.save(os.path.join(args.data_root, 'vclaim.title.npy'), article_embeddings)

  if 'text' in args.input:
    print(verifiedClaims.text)
    logger.info('Getting article vclaim text embeddings')
    article_embeddings = []
    for article_text in tqdm(verifiedClaims.text):
      article_text = nltk.sent_tokenize(article_text)
      if len(article_text) == 0:
        article_embedding = np.array([])
      else:
        article_embedding = encode(article_text, model, tokenizer, config, 'temp')
      article_embeddings.append(article_embedding)
    np.save(os.path.join(args.data_root, 'vclaim.text.npy'), article_embeddings)

  if 'transcript' in args.input:
    for transcript_name in tqdm(dataset.transcripts.keys()):
      transcript = dataset.transcripts[transcript_name]
      transcript = transcript.replace(np.nan, '', regex=True)
      transcript_opath = os.path.join(args.out_dir, '%s.npy'%(transcript_name))
      if os.path.exists(transcript_opath):
        continue
      else:
        transcript_embedding = encode(transcript.sentence, model, tokenizer, config, 'temp')
        np.save(transcript_opath, transcript_embedding)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get elasticsearch scores')
  parser.add_argument('--data-root', '-d', required=True, 
    help='Path to the dataset directory.')
  parser.add_argument('--out-dir', '-o', required=True, 
    help='Path to the output directory')
  parser.add_argument('--config', '-c', required=True, 
    help='Path to the oconfig file')
  parser.add_argument('--input', '-i', nargs='+', 
    default=['vclaim', 'title', 'text', 'transcript'], 
    choices=['vclaim', 'title', 'text', 'transcript'], 
    help='Path to the oconfig file')
  parser.add_argument('--coref-ext', default='', type=str,
    help='If using co-referenced resolved transcripts it gives the resolved coreference data. ({data}/transcripts{EXT} provide EXT)')
  parser.add_argument('--concat-before', default=0, type=int,
    help='Number of sentences concatenated before the input sentence in a transcript')
  parser.add_argument('--concat-after', default=0, type=int,
    help='Number of sentences concatenated after the input sentence in a transcript')

  args = parser.parse_args()
  run(args)
