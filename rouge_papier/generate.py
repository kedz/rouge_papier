from .util import TempFileManager, make_simple_config_text
from . import wrapper
import numpy as np

def compute_extract(sentences, summaries, mode="independent", ngram=1, 
                    length=100, length_unit="word"):

    if mode == "independent":
        return compute_greedy_independent_extract(
            sentences, summaries, ngram, length=length, 
            length_unit=length_unit)
    elif mode == "sequential":
        return compute_greedy_sequential_extract(
            sentences, summaries, ngram, length=length, 
            length_unit=length_unit)
    else:
        raise Exception("mode must be 'independent' or 'sequential'")

def compute_greedy_independent_extract(sentences, summaries, order, 
                                       length=100, length_unit="word"):
    
    with TempFileManager() as manager:
        input_paths = manager.create_temp_files(sentences)
        summary_paths = manager.create_temp_files(summaries)
        config_text = make_simple_config_text([[input_path, summary_paths] 
                                               for input_path in input_paths])
        config_path = manager.create_temp_file(config_text)
        if order == "L":
            df = wrapper.compute_rouge(
                config_path, max_ngram=0, lcs=True, length=length, 
                length_unit=length_unit)
        else:
            order = int(order)
            df = wrapper.compute_rouge(
                config_path, max_ngram=order, lcs=False, length=length, 
                length_unit=length_unit)
            
        scores = df["rouge-{}".format(order)].values.ravel()[:-1]
        ranked_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > 0]

        candidate_extracts = []
        agg_texts = []
        for i in ranked_indices:
            agg_texts.append(sentences[i])
            candidate_extracts.append("\n".join(agg_texts))
        
        input_paths = manager.create_temp_files(candidate_extracts)
        config_text = make_simple_config_text([[input_path, summary_paths] 
                                               for input_path in input_paths])
        config_path = manager.create_temp_file(config_text)
       
        if order == "L":
            df = wrapper.compute_rouge(
                config_path, max_ngram=0, lcs=True, length=length, 
                length_unit=length_unit)
        else:
            df = wrapper.compute_rouge(
                config_path, max_ngram=order, lcs=False, length=length, 
                length_unit=length_unit)
        
        opt_sent_length = np.argmax(
            df["rouge-{}".format(order)].values.ravel()[:-1])
        extract_indices = ranked_indices[:opt_sent_length + 1]
        
        labels = [0] * len(sentences)
        
        for rank, index in enumerate(extract_indices, 1):
            labels[index] = rank
        
        return labels


def compute_greedy_sequential_extract(sentences, summaries, order, 
                                      length=100, length_unit="word"):
    
    with TempFileManager() as manager:
        summary_paths = manager.create_temp_files(summaries)
        
        options = [(i, sent) for i, sent in enumerate(sentences)]

        current_indices = []
        current_summary_sents = []
        current_score = 0

        while len(options) > 0:

            candidates = []
            for idx, sent in options:
                candidates.append("\n".join(current_summary_sents + [sent]))
            candidate_paths = manager.create_temp_files(candidates)

            config_text = make_simple_config_text(
                [[cand_path, summary_paths] for cand_path in candidate_paths])
            config_path = manager.create_temp_file(config_text)

            if order == "L":
                df = wrapper.compute_rouge(
                    config_path, max_ngram=0, lcs=True, length=length, 
                    length_unit=length_unit)
            else:
                order = int(order)
                df = wrapper.compute_rouge(
                    config_path, max_ngram=order, lcs=False, length=length, 
                    length_unit=length_unit)
                
            scores = df["rouge-{}".format(order)].values.ravel()[:-1]
            ranked_indices = [i for i in np.argsort(scores)[::-1]]
            
            if scores[ranked_indices[0]] > current_score:
                current_score = scores[ranked_indices[0]]
                current_indices.append(options[ranked_indices[0]][0])
                current_summary_sents.append(options[ranked_indices[0]][1])
                options.pop(ranked_indices[0])
            else:
                break

        labels = [0] * len(sentences)
        
        for rank, index in enumerate(current_indices, 1):
            labels[index] = rank
        
        return labels
