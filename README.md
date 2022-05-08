# Detecting Previously Fact Checked Claims

This repository contains a dataset and experimental scripts associated the work ["The Role of Context in Detecting Previously Fact-Checked Claims"](https://arxiv.org/abs/2104.07423) accepted at NAACL (Findings), 2022.
The paper focuses on detecting previously fact-checked claims based on the input claims. We focus on claims made in a political debate and we study the impact of modeling the context of the claim: both on the source side, i.e., in the debate, as well as on the target side, i.e., in the fact-checking explanation document.

Code and dataset for a similar work can be found in ["That is a Known Lie: Detecting Previously Fact-Checked Claims"](https://github.com/sshaar/That-is-a-Known-Lie). You can find the paper in this link, https://www.aclweb.org/anthology/2020.acl-main.332.pdf.


__Table of Contents:__
- [Dataset](data/)
- [Experiments](experiments)
  <!-- #- [Fact-Checking-and-Verification-in-Debates](#fact-checking-and-verification-in-debates) -->
  - [Elasticsearch Scores](#elasticsearch-scores)
  - [SBERT Embeddings](#sbert-embeddings)
  - [Features/Scores for the rankSVM](#featuresscores-for-the-ranksvm)
  - [Training the rankSVM](#training-the-ranksvm)
- [Publication](#publication)
- [Credits](#credits)
- [Licensing](#licensing)


## Dataset
More details of the dataset can be found in [data directory](data/).

## Experiments

To train the reranker we need to get several things:
  1. Elasticsearch scores between the input-claims (iclaims) and the verified-claims (vclaims) dataset.
  2. SBERT embeddings of the vclaims and their article.
  3. SBERT embeddings of the iclaims.

After getting them we create the feature vectors for the ranksvm to have a better rerank system.


### Elasticsearch scores
To run elasticsearch you need to first runt he elasticsearch server before running the experiment script.
```
elasticsearch -d # The flag d is to run the elasticsearch server in the background as a demeon process
```

To install elasticsearch, check this [link](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).


After having the elasticsearch server running, to run the experiments you should run
```
data_dir="../data/" #This is the directory with that contains the dataset.
elasticsearch_score_dir="$data_dir/elasticsearch.scores.100" #THe directory where the data will be saved

python get_elasticsearch_scores.py \
		-d $data_dir \
		-o $elasticsearch_score_dir \
		-n 100 \
		--index politifact \
		--coref-ext $coref_ext \
    --load \ # Run thisflag only once to load the data in the elasticsearch server
		--lower 0 --upper 70 --positives
```

Usage of the running script
```
usage: get_elasticsearch_scores.py [-h] --data-root DATA_ROOT --out-dir
                                   OUT_DIR [--nscores NSCORES] [--index INDEX]
                                   [--load] [--positives] [--lower LOWER]
                                   [--upper UPPER]
                                   [--concat-before CONCAT_BEFORE]
                                   [--concat-after CONCAT_AFTER]
                                   [--coref-ext COREF_EXT]

Get elasticsearch scores

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT, -d DATA_ROOT
                        Path to the dataset directory.
  --out-dir OUT_DIR, -o OUT_DIR
                        Path to the output directory
  --nscores NSCORES, -n NSCORES
                        Number of retrieved vclaims scores returned
                        per iclaim.
  --index INDEX         index used to put data in elasticsearch
  --load                Reload the data into elasticsearch if this flag is set.
  --positives           If this flag is set only the sentences with a vclaims
                        would be scored from the transcript
  --lower LOWER         To run the code over batches, the code would run the
                        trasncripts[lower:upper]
  --upper UPPER         To run the code over batches, the code would run the
                        trasncripts[lower:upper]
  --concat-before CONCAT_BEFORE
                        Number of sentences concatenated before the input
                        sentence in a transcript
  --concat-after CONCAT_AFTER
                        Number of sentences concatenated after the input
                        sentence in a transcript
  --coref-ext COREF_EXT
                        If using co-referenced resolved transcripts it gives
                        the resolved coreference data.
                        ({data}/transcripts{COREF_EXT})
```

### SBERT Embeddings
The second step is to get the sentenceBERT (SBERT) embeddings of the following:
  1. vclaim (vclaim)
  2. vclaim-article (title)
  3. vclaim-artcile-title (text)
  4. iclaims (transcript)

To run the experiment,
```
data_dir="../data/politifact"
bert_embedding_dir="$data_dir/SBERT.large.embeddings"

sbert_config="config/bert-specs.json"
python get_bert_embeddings.py \
		-d $data_dir \
		-o $bert_embedding_dir \
		-c $sbert_config \
		--coref-ext $coref_ext \
		-i transcript vclaim title text

```

Usage of the script,
```
usage: get_bert_embeddings.py [-h] --data-root DATA_ROOT --out-dir OUT_DIR
                              --config CONFIG
                              [--input {vclaim,title,text,transcript} [{vclaim,title,text,transcript} ...]]
                              [--coref-ext COREF_EXT]
                              [--concat-before CONCAT_BEFORE]
                              [--concat-after CONCAT_AFTER]

Get elasticsearch scores

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT, -d DATA_ROOT
                        Path to the dataset directory.
  --out-dir OUT_DIR, -o OUT_DIR
                        Path to the output directory
  --config CONFIG, -c CONFIG
                        Path to the config file
  --input {vclaim,title,text,transcript} [{vclaim,title,text,transcript} ...], -i {vclaim,title,text,transcript} [{vclaim,title,text,transcript} ...]
                        What dataentry you want to get the SBERT embeddings for.
  --coref-ext COREF_EXT
                        If using co-referenced resolved transcripts it gives
                        the resolved coreference data.
                        ({data}/transcripts{EXT} provide EXT)
  --concat-before CONCAT_BEFORE
                        Number of sentences concatenated before the input
                        sentence in a transcript
  --concat-after CONCAT_AFTER
                        Number of sentences concatenated after the input
                        sentence in a transcript
```

### Features/Scores for the rankSVM
To create the input for the rankSVM, you combine all the score and recoprical rank you get from elasticssearch when you query on {vclaim, title, text, all} and the cosine similarity score with its recoprical rank between the iclaim and {vclaim, title, top-4 sentences from text}.

To run that script you can do the following,
```
data_dir="../data/politifact"
elasticsearch_score_dir="$data_dir/elasticsearch.scores.100"
bert_embedding_dir="$data_dir/SBERT.large.embeddings"
rerank_embedding_dir="$data_dir/rerank.embeddings"


python get_rerank_embeddings.py \
		-d $data_dir \
		-o $rerank_embedding_dir \
		-b $bert_embedding_dir \
		-e $elasticsearch_score_dir \
		-N 100 \
		-n 4 \
		-m text \ # Use which query mode from elastic search to get the top documents to be reranked
		--lower $1 --upper $2 --positives
```

Usage of the script,
```
usage: get_rerank_embeddings.py [-h] --data-root DATA_ROOT --out-dir OUT_DIR
                                --SBERT-dir SBERT_DIR --elasticsearch-dir
                                ELASTICSEARCH_DIR [--list-length LIST_LENGTH]
                                [--num-sentences NUM_SENTENCES]
                                [--measure {all,vclaim,title,text}]
                                [--lower LOWER] [--upper UPPER]
                                [--num-workers NUM_WORKERS]
                                [--randomness RANDOMNESS] [--manual]
                                [--concat-before CONCAT_BEFORE]
                                [--concat-after CONCAT_AFTER]
                                [--coref-ext COREF_EXT] [--positives]

Get elasticsearch scores

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT, -d DATA_ROOT
                        Path to the dataset directory.
  --out-dir OUT_DIR, -o OUT_DIR
                        Path to the output directory
  --SBERT-dir SBERT_DIR, -b SBERT_DIR
                        Path to the SBERT embedding directory
  --elasticsearch-dir ELASTICSEARCH_DIR, -e ELASTICSEARCH_DIR
                        Path to the elasticsearch embedding directory
  --list-length LIST_LENGTH, -N LIST_LENGTH
                        Number of retrieved verified claims scores returned
                        per sentence.
  --num-sentences NUM_SENTENCES, -n NUM_SENTENCES
                        Number of retrieved verified claims scores returned
                        per sentence.
  --measure {all,vclaim,title,text}, -m {all,vclaim,title,text}
                        Choose ranked list on what metric from elasticsearch
  --lower LOWER         To run the code over batches, the code would run the
                        trasncripts[lower:upper]
  --upper UPPER         To run the code over batches, the code would run the
                        trasncripts[lower:upper]
  --num-workers NUM_WORKERS
                        Number of parallelized processes
  --randomness RANDOMNESS, -r RANDOMNESS
                        Propbablity of adding the true label in the list
  --manual              If the flag is set, then the labels fromt he manual
                        annotations will be considered
  --concat-before CONCAT_BEFORE
                        Number of sentences concatenated before the input
                        sentence in a transcript
  --concat-after CONCAT_AFTER
                        Number of sentences concatenated after the input
                        sentence in a transcript
  --coref-ext COREF_EXT
                        If using co-referenced resolved transcripts it gives
                        the resolved coreference data.
                        ({data}/transcripts{COREF_EXT} provide COREF_EXT)
  --positives           If this flag is set only the sentences with a verified
                        claim would be scored from the transcript
```

### Training the rankSVM
We use the rankSVM tool from [LIBSVM Tools](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#large_scale_ranksvm).
We use the model present in this [zip file](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/ranksvm/liblinear-ranksvm-2.11.zip).
To train the rankSVM, first we will convert the numpy vectors produced by [get_rerank_embeddings.py](experiments/get_rerank_embeddings) to the right format and then train it.

To change the format run,
```
data_dir="../data/politifact"
elasticsearch_score_dir="$data_dir/elasticsearch.scores.100"
bert_embedding_dir="$data_dir/SBERT.large.embeddings"
rerank_embedding_dir="$data_dir/rerank.embeddings"
ranksvm_embeddings_dir=$data_dir/ranksvm.embeddings

python libsvm-datahandler.py \
			-d $data_dir \
			-o $ranksvm_embeddings_dir \
			-r $rerank_embedding_dir \
```

To run the rankSVM, download and MAKE the tool from this [zip file](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/ranksvm/liblinear-ranksvm-2.11.zip).
Then run,
```
data_dir="../data/politifact"
ranksvm_embeddings_dir=$data_dir/ranksvm.embeddings
ranksvm_dir="./ranksvm"

time ./$ranksvm_dir/svm-scale -l -1 -u 1 -s $ranksvm_embeddings_dir/ranksvm.range  $ranksvm_embeddings_dir/train.qid > $ranksvm_embeddings_dir/train.qid.scale
time ./$ranksvm_dir/svm-train $ranksvm_embeddings_dir/train.qid.scale $ranksvm_embeddings_dir/ranksvm.model
time ./$ranksvm_dir/svm-scale -r $ranksvm_embeddings_dir/ranksvm.range $ranksvm_embeddings_dir/test.qid > $ranksvm_embeddings_dir/test.qid.scale
time ./$ranksvm_dir/svm-predict $ranksvm_embeddings_dir/test.qid.scale $ranksvm_embeddings_dir/ranksvm.model $ranksvm_embeddings_dir/test.qid.predict
```


Usage of the script,
```
usage: libsvm-datahandler.py [-h] --data-root DATA_ROOT --out-dir OUT_DIR
                             --reranker-dir RERANKER_DIR [--before BEFORE]
                             [--after AFTER] [--concat-before CONCAT_BEFORE]
                             [--concat-after CONCAT_AFTER]
                             [--coref-ext COREF_EXT]

Get elasticsearch scores

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT, -d DATA_ROOT
                        Path to the dataset directory.
  --out-dir OUT_DIR, -o OUT_DIR
                        Path to the output directory
  --reranker-dir RERANKER_DIR, -r RERANKER_DIR
                        Path to the reranker embeddings directory
  --before BEFORE       Number of sentences you take as context from BEFRORE
                        the sentence
  --after AFTER         Number of sentences you take as context from AFTER the
                        sentence
  --concat-before CONCAT_BEFORE
                        Number of sentences concatenated before the input
                        sentence in a transcript
  --concat-after CONCAT_AFTER
                        Number of sentences concatenated after the input
                        sentence in a transcript
  --coref-ext COREF_EXT
                        If using co-referenced resolved transcripts it gives
                        the resolved coreference data.
                        ({data}/transcripts{EXT} provide EXT)
```

## Publication:
Please cite the following paper.

*Shaden Shaar and Firoj Alam and Da San Martino, Giovanni and
Preslav Nakov, "The Role of Context in Detecting Previously Fact-Checked Claims", Findings of NAACL 2022,  [download](https://arxiv.org/pdf/2104.07423.pdf).*


```bib

@inproceedings{Claim:retrieval:context:2022,
author = {Shaden Shaar and
Firoj Alam and
Da San Martino, Giovanni and
Preslav Nakov},
title = {The Role of Context in Detecting Previously Fact-Checked Claims},
booktitle = {Findings of the Association for Computational Linguistics: NAACL-HLT 2022},
series = {NAACL-HLT~'22},
address = {Seattle, Washington, USA},
year = {2022},
}
```

## Credits
* Shaden Shaar, Qatar Computing Research Institute, HBKU, Qatar
* Firoj Alam, Qatar Computing Research Institute, HBKU, Qatar
* Giovanni Da San Martino, University of Padova, Italy
* Preslav Nakov, Qatar Computing Research Institute, HBKU, Qatar
 Qatar


## Licensing

This dataset is published under CC BY-NC-SA 4.0 license, which means everyone can use this dataset for non-commercial research purpose: https://creativecommons.org/licenses/by-nc/4.0/.
