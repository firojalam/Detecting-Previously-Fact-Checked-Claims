import os
import csv
import numpy as np
import pandas as pd
import pdb
import glob

from logger import logger

VCLAIMS_COLUMNS = ["vclaim_id", "vclaim", "date", "article_url", "truth_meter", 
  "speaker", "title", "text"]
TRANSCRIPT_COLUMNS = ['line_number', 'speaker', 'sentence']
ANNOTATION_COLUMNS = ['line_number', 'sentence', 'vclaim_id', 'vclaims']
FINE_GRAINED_ANNOTATION_COLUMNS = ['line_number', 'sentence', 'vclaim_id', 'vclaims', 'annotation']

class Dataset(object):
  def __init__(self, data_root, transcripts_dir_ext='', concat_before=0, concat_after=0):
    self.data_root = data_root
    
    self.verified_claims = pd.read_csv(os.path.join(data_root, 'verified-claims.tsv'), 
      sep='\t', header=None, names=VCLAIMS_COLUMNS, doublequote=False, quoting=3)

    self.transcripts = {}
    files = glob.glob(os.path.join(data_root, f'transcripts{transcripts_dir_ext}', '*.tsv'))
    files.sort()
    for transcript_fp in files:
      basename = os.path.basename(transcript_fp)[:-4]
      # logger.debug(f'Loading file {basename}')
      
      annotation_fp = os.path.join(data_root, 'fact-checking', basename+'.tsv')
      man_annotation_fp = os.path.join(data_root, 'manual-annotation', basename+'.tsv')
      fine_grained_annotation_fp = os.path.join(data_root, 'fine-grained', basename+'.tsv')
      if not os.path.exists(annotation_fp):
        logger.error(f'There is no annotation file for transcript ({basename})')
        continue

      transcript = pd.read_csv(transcript_fp, sep='\t', header=None, 
        names=TRANSCRIPT_COLUMNS, doublequote=False, quoting=3)
      transcript = transcript.replace(np.nan, '', regex=True)
      if concat_before or concat_after:
        transcript = expand_transcript(transcript, concat_before, concat_after)
      annotation = pd.read_csv(annotation_fp, sep='\t', header=None, 
        names=ANNOTATION_COLUMNS)

      vclaims = np.empty((len(transcript), 0)).tolist()
      
      for index, row in annotation.iterrows():
        vclaims[row.line_number - 1].append(row.vclaim_id)
      transcript['vclaims'] = vclaims
      
      vclaims_man = np.empty((len(transcript), 0)).tolist()
      if os.path.exists(man_annotation_fp):
        man_annotation = pd.read_csv(man_annotation_fp, sep='\t', header=None, 
        names=ANNOTATION_COLUMNS)
        for index, row in man_annotation.iterrows():
          vclaims_man[row.line_number - 1].append(row.vclaim_id)
      transcript['vclaims_man'] = vclaims_man
      
      fine_grained_annotations = [l.copy() for l in transcript['vclaims']]
      if os.path.exists(fine_grained_annotation_fp):
        fine_grained_annotation = pd.read_csv(fine_grained_annotation_fp, sep='\t', header=None, 
        names=FINE_GRAINED_ANNOTATION_COLUMNS)
        for index, row in fine_grained_annotation.iterrows():
          index_ = fine_grained_annotations[row.line_number - 1].index(row.vclaim_id)
          fine_grained_annotations[row.line_number - 1][index_] = row.annotation
        if len(annotation) != len(fine_grained_annotation):
          print(basename)
      transcript['fine_grained'] = fine_grained_annotations

      self.transcripts[basename] = transcript

    self.names = list(self.transcripts.keys()).sort()

  def get_labels(self, transcript, wanted_vclaims_idxs, manual=False):
    logger.info(f'Getting labels for a transcript and manual={manual}')
    labels = np.zeros((len(transcript), wanted_vclaims_idxs.shape[1]))
    for sent_idx in range(len(transcript)):
      for label_idx in range(len(labels[sent_idx])):
        if wanted_vclaims_idxs[sent_idx, label_idx] in transcript.vclaims.iloc[sent_idx]:
          labels[sent_idx, label_idx] = 1
        if manual and wanted_vclaims_idxs[sent_idx, label_idx] in transcript.vclaims_man.iloc[sent_idx]:
          labels[sent_idx, label_idx] = 1
    return labels

  def get_labels_(self, transcript, sentence_idxs, type=''):
    labels = np.zeros((len(sentence_idxs), len(self.verified_claims)))
    for i, sent_idx in enumerate(sentence_idxs):
      for j, label_idx in enumerate(transcript.vclaims[sent_idx]):
        if not type or transcript.fine_grained[sent_idx][j] == type:
          labels[i][label_idx] = 1
          # print('hello')
    return labels

def expand_transcript(transcript, concat_before, concat_after):
  sentences = transcript.sentence.to_list()
  for i in range(concat_before):
    sentences = [''] + sentences
  for i in range(concat_after):
    sentences.append('')
  new_sentences = []
  for i in range(concat_before, len(sentences)-concat_after):
    start = i-concat_before
    end = i+concat_after+1
    sentence = ' '.join(sentences[start:end])
    new_sentences.append(sentence)
  assert len(new_sentences) == len(transcript), f"len of new sentences {len(new_sentences)} is different than len of transcript {len(transcript)}"

  transcript.sentence = new_sentences
  return transcript


if __name__=='__main__':
  d = Dataset('../data/politifact/', before=1, after=1)
  pdb.set_trace()