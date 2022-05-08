import pdb
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from logger import logger
from politifactDataloader import Dataset
np.random.seed(103)


def embed_measure(scores, wanted_vclaims_idxs):
  order = scores.argsort()
  rank = scores.shape[-1] - order.argsort()
  recp_rank = 1/rank

  scores_ = []
  recp_rank_ = []
  for sent_idx in range(len(scores)):
    scores_.append(np.expand_dims(scores[sent_idx, wanted_vclaims_idxs[sent_idx]], axis=0))
    recp_rank_.append(np.expand_dims(recp_rank[sent_idx, wanted_vclaims_idxs[sent_idx]], axis=0))
  
  scores = np.concatenate(scores_, axis=0)
  recp_rank = np.concatenate(recp_rank_, axis=0)

  return scores, recp_rank

flatten = lambda t: [item for sublist in t for item in sublist]


def run(args, lower, upper):
  if not os.path.exists(args.data_root):
    logger.error("Data directory (%s) doesnt exist"%args.data_root)
    exit()
  if not os.path.exists(args.elasticsearch_dir):
    logger.error("elasticsearch scores directory (%s) doesnt exist"%args.elasticsearch_dir)
    exit()
  if not os.path.exists(args.SBERT_dir):
    logger.error("sbert embeddings directory (%s) doesnt exist"%args.SBERT_dir)
    exit()
  if not os.path.exists(os.path.join(args.data_root, 'vclaim.npy')):
    logger.error("sbert embeddings of vclaim doesnt exist")
    exit()
  if not os.path.exists(os.path.join(args.data_root, 'vclaim.title.npy')):
    logger.error("sbert embeddings of vclaim.title doesnt exist")
    exit()
  if not os.path.exists(os.path.join(args.data_root, 'vclaim.text.npy')):
    logger.error("sbert embeddings of vclaim.text doesnt exist")
    exit()
  
  if not os.path.exists(args.out_dir):
    logger.warning("Output directory (%s) doesnt exist"%args.out_dir)
    os.makedirs(args.out_dir)

  dataset = Dataset(args.data_root, args.coref_ext, concat_before=args.concat_before, 
    concat_after=args.concat_after)
  verifiedClaims = dataset.verified_claims

  sbert_vclaims = np.load(os.path.join(args.data_root, 'vclaim.npy'), allow_pickle=True)
  sbert_vclaims_title = np.load(os.path.join(args.data_root, 'vclaim.title.npy'), allow_pickle=True)
  sbert_vclaims_text = np.load(os.path.join(args.data_root, 'vclaim.text.npy'), allow_pickle=True)
  transcript_names = list(dataset.transcripts.keys())
  transcript_names.sort()
  for transcript_name in transcript_names[lower: upper]:
  # for transcript_name in ['20170512_Trump_NBC_holt_interview']:
    logger.info('Working on transcript (%s)'%(transcript_name))
    # Check if the transcript was already encoded
    transcript_opath = os.path.join(args.out_dir, '%s.npz'%(transcript_name))
    transcript = dataset.transcripts[transcript_name]
    if os.path.exists(transcript_opath):
      logger.warning('Transcript encoding already exists at (%s)'%(transcript_opath))
      continue

    logger.info('Encoding transcipt (%s) and outputing it to (%s)'%(transcript_name, transcript_opath))
    sbert_transcript = np.load(os.path.join(args.SBERT_dir, transcript_name+'.npy'), allow_pickle=True)
    elasticsearch_transcript = np.load(os.path.join(args.elasticsearch_dir, transcript_name+'.npz'), allow_pickle=True)

    vclaim_seleced_idxs = elasticsearch_transcript[args.measure][0][:, :args.list_length].astype('int')
    for sent_idx in range(len(vclaim_seleced_idxs)):
      if np.random.choice([0, 1], 1, p=[1-args.randomness, args.randomness])[0]:
        logger.debug('Faking the ranked list and making sure the positives are on top')
        vclaims = transcript.vclaims.iloc[sent_idx]
        current_selected_vclaims = list(vclaim_seleced_idxs[sent_idx])
        for selected_vclaim in vclaims:
          if selected_vclaim in current_selected_vclaims:
            current_selected_vclaims.remove(selected_vclaim)
          # I AM ONLY APPENDING AT THE BEGINNING OF THE LIST 
          # THIS IS HARD CODED!!!!!
          current_selected_vclaims = [selected_vclaim] + current_selected_vclaims
        vclaim_seleced_idxs[sent_idx] = current_selected_vclaims[:args.list_length]


    rerank_transcript = []
    labels = dataset.get_labels(dataset.transcripts[transcript_name], vclaim_seleced_idxs, manual=args.manual)

    # Get embeddings from Elasticsearch scores
    logger.info('Getting scores from elasticsearch')
    for measure in ['vclaim', 'title', 'text']:
      if measure == 'all':
        pass
      idx = elasticsearch_transcript[measure][0].astype('int')
      small_scores = elasticsearch_transcript[measure][1]
      
      scores = np.zeros((len(small_scores), len(verifiedClaims)))
      for sent_idx, (vclaim_idxs, vclaim_scores) in enumerate(zip(idx, small_scores)):
        scores[sent_idx][vclaim_idxs] = vclaim_scores
      
      scores, recp_rank = embed_measure(scores, vclaim_seleced_idxs)
      
      scores = np.expand_dims(scores, axis=2)
      recp_rank = np.expand_dims(recp_rank, axis=2)
      rerank_transcript.append(scores)
      rerank_transcript.append(recp_rank)

    positives_indices_ = np.arange(len(transcript))
    if args.positives:
      positives_indices_ = []
      [positives_indices_.append(i) if len(x) else None for i, x in enumerate(transcript.vclaims)]
      positives_indices_ = np.array(positives_indices_)
    positives_indices = flatten([list(range(elm-3, elm+2)) for elm in positives_indices_])
    positives_indices = list(set(positives_indices))
    positives_indices.sort()
    print(positives_indices)

    # Get embeddings from SBERT embeddings of vclaim and vclaim_title
    logger.info('Getting scores from sbert_vclaims and sbert_vclaims_title')
    for sbert_embeddings in [sbert_vclaims, sbert_vclaims_title]:
      scores = np.zeros((len(sbert_transcript), len(verifiedClaims)))
      scores[positives_indices] = metrics.pairwise.cosine_similarity(sbert_transcript[positives_indices], sbert_embeddings)
      scores, recp_rank = embed_measure(scores, vclaim_seleced_idxs)
      scores = np.expand_dims(scores, axis=2)
      recp_rank = np.expand_dims(recp_rank, axis=2)
      rerank_transcript.append(scores)
      rerank_transcript.append(recp_rank)

    # Get embeddings from SBERT embeddings of vclaim_text
    logger.info('Getting scores from sbert_vclaims_text_scores')
    sbert_vclaims_text_scores = np.zeros((len(sbert_transcript), args.num_sentences, len(verifiedClaims)))
    for vclaim_id, sbert_embeddings in enumerate(tqdm(sbert_vclaims_text)):
      if not len(sbert_embeddings):
        continue
      # sbert_embeddings = sbert_embeddings.squeeze()
      vclaim_text_score = metrics.pairwise.cosine_similarity(sbert_transcript[positives_indices], sbert_embeddings)
      vclaim_text_score = np.sort(vclaim_text_score)
      n = min(args.num_sentences, len(sbert_embeddings))
      sbert_vclaims_text_scores[positives_indices, :n, vclaim_id] = vclaim_text_score[:, -n:]
    for n in range(args.num_sentences):
      scores = sbert_vclaims_text_scores[:, n, :]
      scores, recp_rank = embed_measure(scores, vclaim_seleced_idxs)
      scores = np.expand_dims(scores, axis=2)
      recp_rank = np.expand_dims(recp_rank, axis=2)
      rerank_transcript.append(scores)
      rerank_transcript.append(recp_rank)

    rerank_transcript = np.concatenate(rerank_transcript, axis=2)
    logger.info(f'Finished encoding transcript ({transcript_name}) with shape ({rerank_transcript.shape}) and saving it ({transcript_opath})')
    np.savez(transcript_opath, input=rerank_transcript, labels=labels)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get elasticsearch scores')
  parser.add_argument('--data-root', '-d', required=True, 
    help='Path to the dataset directory.')
  parser.add_argument('--out-dir', '-o', required=True, 
    help='Path to the output directory')
  parser.add_argument('--SBERT-dir', '-b', required=True, 
    help='Path to the SBERT embedding directory')
  parser.add_argument('--elasticsearch-dir', '-e', required=True, 
    help='Path to the elasticsearch embedding directory')
  parser.add_argument('--list-length', '-N', default=100, type=int,
    help='Number of retrieved verified claims scores returned per sentence.')
  parser.add_argument('--num-sentences', '-n', default=4, type=int,
    help='Number of retrieved verified claims scores returned per sentence.')
  parser.add_argument('--measure', '-m', default='text',
    choices=['all', 'vclaim', 'title', 'text'],
    help='Choose ranked list on what metric from elasticsearch')
  parser.add_argument('--lower', default=0, type=int,
    help='faster parallelization')
  parser.add_argument('--upper', default=100, type=int,
    help='faster parallelization')
  parser.add_argument('--num-workers', default=4, type=int,
    help='Number of parallelized processes')
  parser.add_argument('--randomness', '-r', default=0, type=float,
    help='Propbablity of adding the true label in the list')
  parser.add_argument('--manual', default=False, action='store_true',
    help='If the flag is set, then the labels fromt he manual annotations will be considered')
  parser.add_argument('--concat-before', default=0, type=int,
    help='Number of sentences concatenated before the input sentence in a transcript')
  parser.add_argument('--concat-after', default=0, type=int,
    help='Number of sentences concatenated after the input sentence in a transcript')
  parser.add_argument('--coref-ext', default='', type=str,
    help='If using co-referenced resolved transcripts it gives the resolved coreference data. ({data}/transcripts{EXT} provide EXT)')
  parser.add_argument('--positives', default=False, action='store_true',
    help='If this flag is set only the sentences with a verified claim would be scored from the transcript')
  args = parser.parse_args()
  run(args, args.lower, args.upper)

