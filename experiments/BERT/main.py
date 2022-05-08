import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

import utils
from logger import logger

import torch 
import transformers 
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_path):
    logger.info('Encoding using (%s)'%model_path)
    if 'sts' in model_path:
        return (SentenceTransformer(model_path).to(DEVICE), None)
    elif 'albert' in model_path:
        return (transformers.AlbertModel, transformers.AlbertTokenizer)
    elif 'roberta' in model_path:
        return (transformers.RobertaModel, transformers.RobertaTokenizer)
    elif 'bert' in model_path:
        return (transformers.BertModel, transformers.BertTokenizer)
    elif 'openai-gpt' == model_path:
        return (transformers.OpenAIGPTModel, transformers.OpenAIGPTTokenizer)
    elif 'gpt2' == model_path:
        return (transformers.GPT2Model, transformers.GPT2Tokenizer)
    elif 'xlnet' in model_path:
        return (transformers.XLNetModel, transformers.XLNetTokenizer)
    elif 'xlm' in model_path:
        return (transformers.XLMModel, transformers.XLMTokenizer)


def encode_transformers(sentences, model_class, tokenizer_class, pretrained_weights, 
           out_file='', pooling_layer=-2, pooling_strategy='reduce_mean'):
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    model.to(DEVICE)
    model.eval()

    logger.info('Running (%s) on (%d) sentences'%(pretrained_weights, len(sentences)))
    sentences_vector = []
    for sentence in tqdm(sentences):
        
        # NOTE: pytorch_transforms doesnt add the the starting and seperating 
        #           tokens so we need to add that to the sentence manually. 
        input = torch.tensor([tokenizer.encode('%s %s %s'%(cls_token, sentence, sep_token))]).to(DEVICE)    
        
        # Since the model output all the hidden layers output, we get that output 
        #   by indexing to [2]. We then get usee the pooling layer desired to get
        #   the exact output we want. 
        sentence_vector = model(input)[2][pooling_layer]

        # Mount bacK to CPU if needed
        if torch.cuda.is_available():
            sentence_vector = sentence_vector.cpu()

        # Pool
        sentence_vector = sentence_vector.detach().numpy()
        sentence_vector = pooling(sentence_vector, pooling_strategy)

        sentences_vector.append(sentence_vector)

    sentences_vector = np.concatenate(sentences_vector, axis=0)

    # Save the embeddings as a numpy file if wanted
    if out_file:
        logger.info('Embeddings are being exported to (%s)'%out_file)
        np.save(out_file, sentences_vector)

    return sentences_vector


# Mask the input with the appropriate pooling_strategy. 
# Input is a numpy array and the Output is a numpy array.
def pooling(input, pooling_strategy):

    mask = np.ones((input.shape[0], input.shape[1]))

    if pooling_strategy == 'reduce-mean':
        output = masked_reduce_mean(input, mask)
    elif pooling_strategy == 'reduce-max':
        output = masked_reduce_max(input, mask)
    elif pooling_strategy == 'reduce-mean-max':
        output = np.concatenate([masked_reduce_mean(input, mask),
                                 masked_reduce_max(input, mask)], axis=1)

    return output


# Mask functions obtained from bert-as-a-service.
def minus_mask(x, m):
    return x - np.expand_dims(1.0 - m, axis=-1) * 1e30

def mul_mask(x, m):
    return x * np.expand_dims(m, axis=-1)

def masked_reduce_max(x, m):
    return np.max(minus_mask(x, m), axis=1)

def masked_reduce_mean(x, m):
    return np.sum(mul_mask(x, m), axis=1) / (np.sum(m, axis=1, keepdims=True) + 1e-10)



def encode_sentence_bert(sentences, model, out_file='', show_progress_bar=False):

    # logger.info('Encoding (%d) sentences'%len(sentences))
    embeddings = model.encode(sentences, show_progress_bar=show_progress_bar, batch_size=1024)
    # logger.info('Concatenating (%d) sentences'%len(sentences))
    embeddings = [np.expand_dims(embed, axis=0) for embed in embeddings]
    embeddings = np.concatenate(embeddings)

    # Save the embeddings as a numpy file if wanted
    if out_file:
        logger.info('Embeddings are being exported to (%s)'%out_file)
        np.save(out_file, embeddings)

    return embeddings


def get_scores(claims_embeddings, facts_embeddings):

    scores = np.empty((len(claims_embeddings), facts_embeddings.shape[0]))
    
    MAP_scores = []

    frequency = np.zeros(facts_embeddings.shape[0])
    
    logger.info("Evaluation")
    for i, sentence in enumerate(tqdm(claims_embeddings)):
        
        sentence_embedding = claims_embeddings[i]
        sentence_embedding = sentence_embedding.reshape((1, -1))
        query_score =  metrics.pairwise.cosine_similarity(facts_embeddings, sentence_embedding).squeeze()
        scores[i] = query_score

    return scores


