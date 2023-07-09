from glob import glob
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, LEDForConditionalGeneration,
                          T5Tokenizer, T5ForConditionalGeneration,
                          BartForConditionalGeneration, BartTokenizer)
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import logging
import json
import glob
import ntpath
import string
import pickle
import sys
import os
import nltk
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
os.environ['DATA_RANGE'] = sys.argv[2]
os.environ['AGGREGATOR'] = sys.argv[3]
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import benepar
import random
import math
import time
import multiprocessing
import T5_QG
from nltk.tokenize import sent_tokenize, word_tokenize
from graph_modules import Graph
from modules import Token, Span
from typing import List, Dict, Tuple, Set, Mapping
from utils import convert_qasper_context, load_allennlp_ckpt
import hashlib


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filename='span_collecto_v3.log', filemode='w',
                    level=logging.INFO)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_md5(paper_id, sec_id, para_id, question, answer_text):
    raw_string = str(paper_id) + str(sec_id) + str(para_id) + question + answer_text
    return hashlib.md5(raw_string.encode()).hexdigest()  

  
set_seed(42)

bene_parser = benepar.Parser("benepar_en3")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
archive_file = './qasper-led-baseline/output_full_ulqa_v2.1/model.tar.gz'
model = load_allennlp_ckpt(archive_file)
model_name = 'allenai/led-base-16384'
# attention_window_size=768 # 64
attention_window_size = model.config.attention_window[0]
stop_words = set(stopwords.words('english'))
stop_spans = {'Related Work', 'Related Works', '[ PAD ] .'}
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = model.led.encoder
encoder.eval()
encoder.to(device1)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_model.to(device)
t5_model.eval()

all_models = [model, t5_model]
if os.environ['AGGREGATOR'] == 'bart':
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    bart_model.to(device)
    bart_model.eval()
    all_models.append(bart_model)
for cur_model in all_models:
    for param in cur_model.parameters():
        param.requires_grad = False

t5_nlp = T5_QG.pipeline("question-generation", model='valhalla/t5-base-qg-hl', qg_format="highlight",
                        gpu_index=0)


def compile_id2token(tokens: List[Token], inputs):
    id2token = dict((t_idx, t) for t_idx, t in enumerate(tokens))
    id2tokenId = dict((t_idx, t.item()) for t_idx, t in enumerate(inputs[0]))
    return id2token, id2tokenId


def check_prefix_space(token: str):
    """

    Args:
        token (str): _description_
    """
    return not (ord(token[0]) >= 0 and ord(token[0]) <= 127)


def select_head(attention: torch.Tensor, id2token: Dict, tokens: List[Token], Spans) -> List[List[Token]]:
    head_nums = 12
    layer_nums = 6
    max_span_len = 80
    assert max_span_len <= attention_window_size // 2
    special_tokens = ['<s>', '<pad>', '</s>', '<unk>']
    logging.info(attention[0].size())
    right_attention_start_pos = attention_window_size // 2
    max_token_num = len(id2token)
    layer_head_to_spans = {}
    layer_head_to_raw_spans = {}
    max_match_num = -1
    if len(tokens) == 1:
        spans = [tokenizer.convert_tokens_to_string([token.s for token in tokens]).strip().lower()]
        return spans, [tokens]
    for l_idx in range(layer_nums):
        for h_idx in range(head_nums):
            spanized_seq = []
            t_len = len(tokens)
            t_idx = 0
            while t_idx < t_len:
                token = tokens[t_idx]
                token_s = token.s
                if t_idx == t_len - 1:
                    spanized_seq.append([token])
                    t_idx += 1
                    continue
                if token_s in special_tokens:
                    spanized_seq.append([token])
                    t_idx += 1
                    continue
                if token_s.replace('Ġ', '').lower() in stop_words and (t_idx == len(id2token) - 1 or check_prefix_space(id2token[t_idx + 1].s)): # 停用词单独作为一个span
                    spanized_seq.append([token])
                    t_idx += 1
                    continue
                assert check_prefix_space(token_s) or t_idx == 0, 't_idx: [{}], token: [{}], tokens: {}'.format(
                    t_idx, token, tokens)

                if t_idx + max_span_len + 1 - max_token_num > 0:
                    end_relative_pos = right_attention_start_pos + max_token_num - t_idx
                else:
                    end_relative_pos = right_attention_start_pos + max_span_len + 1
                candidate_attention_scores = attention[l_idx][0][h_idx][t_idx][right_attention_start_pos + 1:end_relative_pos].cpu()
                candidate_ids = list(range(t_idx + 1, t_idx + end_relative_pos - right_attention_start_pos))
                if len(candidate_ids) == 0:
                    print('t_idx: [[{}]], end_relative_pos: [[{}]], right_attention_start_pos: [[{}]]'.format(
                        t_idx, end_relative_pos, right_attention_start_pos
                    ))
                    print('cur token: [{}], len: {}'.format(tokens[t_idx], len(tokens[t_idx].s)))
                    print('processed token: [{}], len: {}'.format(tokens[t_idx].s.replace('Ġ', ''),
                                                                len(tokens[t_idx].s.replace('Ġ', ''))))
                    print('is stopword: [{}]'.format(tokens[t_idx].s.replace('Ġ', '').lower() in stop_words))
                new_attention_scores = []
                new_candidate_ids = []
                max_score = -1e10
                max_index = -1
                for1_flag = False
                if1_flag = False
                if2_flag = False
                for attention_score, index in zip(candidate_attention_scores, candidate_ids):
                    token = id2token[index]
                    _token_s = token.s
                    for1_flag = True
                    if not check_prefix_space(_token_s) and index != 0 or \
                        (index == len(id2token) - 1 and check_prefix_space(_token_s)) or \
                        (check_prefix_space(_token_s) and check_prefix_space(id2token[index + 1].s)):
                        if1_flag = True
                        attention_score = attention_score.item()
                        assert isinstance(attention_score, float)
                        if attention_score > max_score and (index == len(id2token) - 1 or check_prefix_space(id2token[index + 1].s)):
                            if2_flag = True
                            max_score = attention_score
                            max_index = index
                assert max_index != -1, 't_idx: [{}], end_relative_pos: [{}], right_attention_start_pos: [{}], max_span_len: [{}], max_token_num: [{}], token_s: [{}], cur_token_s: [{}], next_token_s: [{}], flags: {}, index: {}, attention_score: {}, attention: {}; candidate_ids: {}; tokens: {}'.format(
                    t_idx,
                    end_relative_pos,
                    right_attention_start_pos,
                    max_span_len,
                    max_token_num,
                    token_s,
                    id2token[index].s,
                    id2token[index + 1].s if index + 1 < len(id2token) else 'None',
                    (for1_flag, if1_flag, if2_flag),
                    index,
                    attention_score,
                    attention[0].size(),
                    candidate_ids,
                    tokens)
                spanized_seq.append(tokens[t_idx: max_index + 1])
                t_idx = max_index + 1
            spans = [tokenizer.convert_tokens_to_string([token.s for token in seq]).strip().lower()
                     for seq in spanized_seq]
            layer_head_to_spans['layer-{},head-{}'.format(l_idx, h_idx)] = spans
            layer_head_to_raw_spans['layer-{},head-{}'.format(l_idx, h_idx)] = spanized_seq
            spans = set(spans)
            match_num = len(Spans & spans)
            if match_num > max_match_num:
                max_match_head = 'layer-{},head-{}'.format(l_idx, h_idx)
                max_match_num = match_num
    return layer_head_to_spans[max_match_head], layer_head_to_raw_spans[max_match_head]


def span_filter(spans):
    """Filter some trivial or redundant spans with specific rules"""
    filtered_spans = []
    for span in spans:
        assert isinstance(span, list)
        assert isinstance(span[0], tuple)
        assert isinstance(span[0][0], str)
        assert isinstance(span[0][1], list)
        assert isinstance(span[0][1][0], Token)
        assert len(span) > 0
        span = tuple([(item[0], tuple(item[1])) for item in span])
        union_set = set()
        token_sets = []
        token_list = []
        new_span = []
        for subspan in span:
            tokens = word_tokenize(subspan[0])
            if len(tokens) == 0:
                continue
            trivial_inspan_token_num = 0
            for t in tokens:
                if t in string.punctuation or t.lower() in stop_words:
                    trivial_inspan_token_num += 1
            if trivial_inspan_token_num / len(tokens) >= 0.5:
                continue
            new_span.append(subspan)
            token_list += tokens
            token_sets.append(set(tokens))
        if len(token_list) == 0:
            continue
        trivial_token_num = 0
        for token in token_list:
            if token in string.punctuation or token.lower() in stop_words:
                trivial_token_num += 1
        if trivial_token_num / len(token_list) > 0.5:
            continue
        for token_set in token_sets:
            union_set |= token_set
        token_num = sum([len(token_set) for token_set in token_sets])
        nonoverlap_rate = len(union_set) / token_num
        if nonoverlap_rate > 0.8:
            filtered_spans.append(tuple(new_span))
    filtered_spans = list(set(filtered_spans))
    filtered_spans = [list(item) for item in filtered_spans]
    print('remaining span num: [{}]'.format(len(filtered_spans)))
    return filtered_spans


def answer_aggregator_via_t5(spans):
    """aggregate each set of nodes into a spans with T5 model"""
    connected_spans = []
    for span in tqdm(spans, desc='aggregate spans'):
        extra_id_num = len(span) - 1
        span_with_mask = []
        for s_idx, subspan in enumerate(span):
            span_with_mask.append(subspan[0])
            span_with_mask.append('<extra_id_{}>'.format(s_idx))
        span_with_mask = span_with_mask[:-1]
        input_ids = t5_tokenizer(' '.join(span_with_mask), return_tensors='pt').to(device).input_ids
        sequence_ids = t5_model.generate(input_ids, do_sample=False, no_repeat_ngram_size=2, num_beams=5,
                                         max_length=max(input_ids.size(1) + 3 * extra_id_num, input_ids.size(1) + 1))
        sequence = t5_tokenizer.batch_decode(sequence_ids)[0].replace('</s>', '')
        mask_id_to_txt = {}
        span_start_idx = 0
        bad_span = False
        for s_idx in range(len(span) - 1):
            mask_id = '<extra_id_{}>'.format(s_idx)
            mask_len = len(mask_id)
            mask_start_idx = sequence.find(mask_id)
            if mask_start_idx == -1:
                bad_span = True
                break
            span_start_idx += mask_start_idx + mask_len
            if s_idx <= len(span) - 2:
                next_mask_id = '<extra_id_{}>'.format(s_idx + 1)
                span_end_idx = sequence.find(next_mask_id)
                if span_end_idx == -1:
                    bad_span = True
                    break
                mask_id_to_txt[mask_id] = sequence[span_start_idx: span_end_idx].strip()
            else:
                next_mask_id = '<extra_id_{}>'.format(s_idx + 1)
                span_end_idx = sequence.find(next_mask_id)
                if span_end_idx == -1:
                    mask_id_to_txt[mask_id] = sequence[span_start_idx:].strip()
                else:
                    mask_id_to_txt[mask_id] = sequence[span_start_idx: span_end_idx].strip()
        if bad_span is True:
            continue
        filled_span_with_mask = []
        for _span in span_with_mask:
            if _span.startswith('<extra_id_'):
                filled_span_with_mask.append(mask_id_to_txt[_span])
            else:
                filled_span_with_mask.append(_span)
        connected_spans.append(Span(s=' '.join(filled_span_with_mask),
                                    tokens=list((item[0], list(item[1])) for item in span)))
    print('The number of connected_spans: [{}]'.format(len(connected_spans)))
    return connected_spans


def answer_aggregator_via_bart(spans):
    """aggregate each set of nodes into a spans with BART model"""
    connected_spans = []
    connect_cnt = 0
    for span in tqdm(spans, desc='aggregate spans'):
        extra_id_num = len(span) - 1
        span_with_mask = []
        used_spans = set()
        for s_idx, subspan in enumerate(span):
            # do not consider duplicated spans in answer aggregation (but still consider its context sentence)
            if subspan[0].lower().strip() in used_spans:
                continue
            used_spans.add(subspan[0].lower().strip())
            span_with_mask.append(subspan[0].strip())
            span_with_mask.append('<mask>')
        span_with_mask = span_with_mask[:-1]
        if len(span_with_mask) > 1:
            connect_cnt += 1
            batch = bart_tokenizer(' '.join(span_with_mask), return_tensors="pt").to(device)
            generated_ids = bart_model.generate(batch["input_ids"])
            span_str = bart_tokenizer.batch_decode(generated_ids, no_repeat_ngram_size=3,
                                                   num_beams=5, max_length=max(batch["input_ids"].size(1) + 3 * extra_id_num,
                                                                               batch["input_ids"].size(1) + 1),
                                                   skip_special_tokens=True)[0]
        else:
            span_str = span_with_mask[0]
        connected_spans.append(Span(s=span_str,
                                    subspans=[subspan[0] for subspan in span],
                                    tokens=list((item[0], list(item[1])) for item in span)))
    print('The number of connected_spans: [{}]'.format(len(connected_spans)))
    print('The number need to connect spans: [{}]'.format(connect_cnt))
    return connected_spans


def combine_questions(short_qas):
    """

    Args:
        short_qas (_type_): _description_
    """
    pass


def span_to_qa_pair(sents, spans):
    """

    Args:
        sents (List[str]): _description_
        spans (List[Span]): _description_
    """
    sent_bows = [set(word_tokenize(sent)) for sent in sents]
    generated_qas = []
    for span in tqdm(spans, desc='generate qas...'):
        short_qas = []
        for item in span.tokens:
            txt = item[0]
            txt_bow = set(word_tokenize(txt))
            sent_ids = [it.sent_id for it in item[1]]
            assert sent_ids[0] == sent_ids[-1]
            sent_idx = sent_ids[0]
            sent = sents[sent_idx]
            sent_bow = sent_bows[sent_idx]
            assert txt_bow.issubset(sent_bow), 'span: {}; sent: {}'.format(txt, sent)
            qa = t5_nlp.qg_with_answer_text(sent, txt)
            short_qas.append(qa)
        span.short_qas = short_qas
    

def direct_to_long_question(sents, spans):
    """

    Args:
        sents (_type_): _description_
        spans (_type_): _description_
    """
    # sent_bows = [set(word_tokenize(sent)) for sent in sents]
    generated_qas = []
    for span in tqdm(spans, desc='directly generate qas...'):
        short_qas = []
        span_level_sents = set()
        for item in span.tokens:
            span_located_ids = [(it.sec_idx, it.para_idx, it.sent_id) for it in item[1]]
            assert span_located_ids[0] == span_located_ids[-1]
            span_level_sents.add(span_located_ids[0])
        answer_txt = span.s
        span_level_sents = sorted(list(span_level_sents))
        context = ' '.join([sents[sec_idx][para_idx][sent_id] for sec_idx, para_idx, sent_id in span_level_sents])
        long_qa = t5_nlp.qg_with_answer_text(context, answer_txt)
        span.long_qas = [long_qa]


def collect_span_in_specific_head(paper_id, Final_raw_spans, trivial_span_ids, full_text_attention,
                                  attention_window_size, _tokenizer, global_idx, global_span_ids=None,
                                  global_attention_l2r=None, global_attention_r2l=None):
    final_spans = []
    has_global_edges = []
    assert global_attention_l2r is not None and global_attention_r2l is not None and global_span_ids is not None
    for text_attention, ga_l2r, ga_r2l in zip(full_text_attention, global_attention_l2r,
                                              global_attention_r2l):
        g = Graph(seq=Final_raw_spans, trivial_span_ids=trivial_span_ids,
                  half_window_size=attention_window_size // 2,
                  attention=text_attention, tokenizer=_tokenizer,
                  global_span_ids=global_span_ids,
                  ga_l2r=ga_l2r, ga_r2l=ga_r2l)
        g.parse_seq()
        g.build_edges()
        g.calculate_edge_values()
        g.update_edge_values_via_global_attention(top_k=3)
        g.span_collector_via_dfs(threshold=0.4)
        spans = g.get_graph_clustered_spans()
        all_cur_spans = spans['l2r'] + [item[::-1] for item in spans['r2l']]
        flatten_all_cur_spans = []
        flatten_is_including_global_edges = []
        for span_idx, span in enumerate(all_cur_spans):
            s_idx = -1
            cur_span = []
            assert span is not None, '{}'.format(spans)
            is_including_global_edges = False
            for item_idx, item in enumerate(span):
                assert isinstance(item[0], int)
                assert isinstance(item[1], str)
                assert isinstance(item[2][0], Token)
                assert s_idx < item[0]
                s_idx = item[0]
                cur_span.append((item[1].strip(), item[2]))
                if item_idx > 0:
                    tmp_edge = (span[item_idx - 1][0], span[item_idx][0])
                    if tmp_edge in g.valuable_global_edges:
                        is_including_global_edges = True

            if len(cur_span) > 0:
                flatten_all_cur_spans.append(cur_span)
                flatten_is_including_global_edges.append(is_including_global_edges)
        if len(flatten_all_cur_spans) > 0:
            final_spans += flatten_all_cur_spans
            has_global_edges += flatten_is_including_global_edges
    assert len(final_spans) == len(has_global_edges)
    cached_path = 'final_spans_{}_{}.pkl'.format(paper_id, global_idx)
    with open(cached_path, 'wb') as f:
        pickle.dump({'final_spans': final_spans, 'has_global_edges': has_global_edges}, f)
    return cached_path


def span_collect_with_multi_processes(paper_id, layer_nums, head_nums, Final_raw_spans,
                                      trivial_span_ids,
                                      full_text_attention, attention_window_size,
                                      _tokenizer,
                                      global_span_ids=None,
                                      global_attention_l2r=None,
                                      global_attention_r2l=None):
    num_threads = 12
    pool = []
    assert global_attention_l2r is not None and global_attention_r2l is not None
    p = multiprocessing.Pool(num_threads)
    lh_pairs = [(l_idx, h_idx) for l_idx in range(layer_nums) for h_idx in range(head_nums)]
    for i in range(num_threads):
        start_index = len(lh_pairs) // num_threads * i
        end_index = len(lh_pairs) // num_threads * (i + 1)
        if i == num_threads - 1:
            end_index = len(lh_pairs)
        pool.append(p.apply_async(collect_span_in_specific_head, args=(
            paper_id, Final_raw_spans, trivial_span_ids,
            [full_text_attention[l_idx][0][h_idx] for l_idx, h_idx in lh_pairs[start_index: end_index]],
            attention_window_size, _tokenizer, i, global_span_ids,
            [global_attention_l2r[l_idx][0][h_idx] for l_idx, h_idx in lh_pairs[start_index: end_index]],
            [global_attention_r2l[l_idx][0][h_idx] for l_idx, h_idx in lh_pairs[start_index: end_index]])))
    p.close()
    p.join()

    Final_spans = []
    Final_has_global_edges = []
    for i, thread in enumerate(pool):
        cached_path_tmp = thread.get()
        logging.info("Reading thread {} output from {}".format(i, cached_path_tmp))
        with open(cached_path_tmp, "rb") as reader:
            features_tmp = pickle.load(reader)
        os.remove(cached_path_tmp)
        Final_spans += features_tmp['final_spans']
        Final_has_global_edges += features_tmp['has_global_edges']
    return Final_spans, Final_has_global_edges


def heuristic_filter(qas: List[Dict], is_global: List[bool]):
    """

    Args:
        qas (List[Dict]): _description_
    """
    filtered_qas = []
    filtered_is_global = []
    for qa, _global in zip(qas, is_global):
        Q = set(qa['question'].split())
        A = set(qa['answer'].split())
        if Q & A == A:
            continue
        filtered_qas.append(qa)
        filtered_is_global.append(_global)
    return filtered_qas, filtered_is_global


def write_to_file(input_file, cur_paper_id, paper_id_to_paras, paper_sentnized, output_qas, output_path,
                  is_global):
    """

    Args:
        input_file (_type_): _description_
        paper_id_to_paras (_type_): _description_
        output_qas (_type_): _description_
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        qasper = json.load(f)
    paper_id_to_qas = {}
    qids = set()
    for qa, _global in zip(output_qas, is_global):
        tokens = qa.tokens[0][1]
        paper_id = tokens[0].paper_id
        sec_para_ids = set()
        sents = set()
        for tokens in qa.tokens:
            for t in tokens[1]:
                sec_id = t.sec_idx
                para_id = t.para_idx
                sent_id = t.sent_id
                sp_tuple = (sec_id, para_id)
                try:
                    sents.add(paper_sentnized[sec_id][para_id][sent_id])
                except Exception as e:
                    print(e)
                    with open('output_qas.pkl', 'wb') as f:
                        pickle.dump(output_qas, f)
                    with open('paper_sentnized.pkl', 'wb') as f:
                        pickle.dump(paper_sentnized, f)
                    print(sec_id, para_id, sent_id)
                    exit(0)
                sec_para_ids.add(sp_tuple)
        sec_para_ids = sorted(list(sec_para_ids))

        full_text = qasper[paper_id]['full_text']
                    
        long_qa = qa.long_qas[0][0]
        subspans = qa.subspans
        question = long_qa['question']
        if not question.endswith('?'):
            continue
        answer = long_qa['answer']
        highlighted_evidence = list(sents)
        evidence = []
        for sp in sec_para_ids:
            sec_id = sp[0]
            para_id = sp[1]
            sec = full_text[sec_id]
            if para_id == 0:
                evidence.append(sec['section_name'])
            else:
                evidence.append(sec['paragraphs'][para_id - 1])
        final_qa = {
            'question': question,
            'question_id': get_md5(paper_id, '', '', question, answer),
            'answers':[
                {
                    'answer': {
                        'unanswerable': False,
                        'extractive_spans': [],
                        'yes_no': None,
                        'free_form_answer': answer,
                        'evidence': evidence,
                        'highlighted_evidence': highlighted_evidence,
                        'subspans': subspans,
                        'is_global': _global
                    }
                }
            ]
        }
        if final_qa['question_id'] not in qids:
            qids.add(final_qa['question_id'])
            if paper_id not in paper_id_to_qas:
                paper_id_to_qas[paper_id] = []
            paper_id_to_qas[paper_id].append(final_qa)

    for paper_id, paper in qasper.items():
        if paper_id in paper_id_to_qas:
            paper['qas'] = paper_id_to_qas[paper_id]

    if len(output_qas) > 0:
        assert len(paper_id_to_qas) == 1, '**{}**'.format(paper_id_to_qas)
        the_only_paper_id = list(paper_id_to_qas.keys())[0]
        assert the_only_paper_id in output_path
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({the_only_paper_id: qasper[the_only_paper_id]}, f, ensure_ascii=False, indent=2)
    else:
        paper = qasper[cur_paper_id]
        paper['qas'] = []
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({cur_paper_id: paper}, f, ensure_ascii=False, indent=2)


def find_answer_span_via_type(start_idx: int,
                              tree_node: nltk.Tree,
                              sparse_graph: Mapping[Tuple[int, int], List[Tuple[int, int]]],
                              valuable_parents: List[Tuple[int, int]] = []) -> List[Tuple[str, str, Tuple[int, int]]]:
    """search spans from a sentence(parsed as a tree_node) and return the
        proper spans with type "xP" / "SBAR"

    Args:
        start_idx: the start index in this sentence. This parameter is used to
            collect the position of the span in the sentence.
        tree_node (nltk.Tree): a parsed sentence
        sparse_graph: save edges between each pair of potential spans (e.g. NP, ADJP, VP, SBAR)
        valuable_parents: maintain parent span list, where each parent must be a potential span

    Returns:
        list: a list of the satisfying spans, their positions and types
    """
    spans = []
    if isinstance(tree_node, str):
        return spans
    label = tree_node.label()
    span_tokens = tree_node.leaves()
    desired_types = {'NP', 'VP', 'ADJP', 'SBAR'}
    # skip stop_words and stop_spans
    add_flag = False
    if (label in desired_types and ' '.join(span_tokens) not in stop_spans 
        and (len(span_tokens) > 1 or span_tokens[0] not in stop_words)):
        # the range of span: [span_start_idx, span_end_idx)
        span_start_idx = start_idx
        span_end_idx = span_start_idx + len(span_tokens)
        cur_pos_range = (span_start_idx, span_end_idx)
        # each tuple in span: (span_str, span_label, [span_start_idx, span_end_idx])
        # [span_start_idx, span_end_idx)
        spans.append((' '.join(span_tokens), label, (span_start_idx, span_end_idx)))
        if cur_pos_range not in sparse_graph:
            sparse_graph[cur_pos_range] = []
            if len(valuable_parents) > 0: # build edges between interdependent spans
                for pos_range in valuable_parents:
                    sparse_graph[pos_range].append(cur_pos_range)
                    sparse_graph[cur_pos_range].append(pos_range)
            valuable_parents.append(cur_pos_range)
            add_flag = True

    child_start = start_idx
    for child in tree_node:
        if not isinstance(child, str):
            spans += find_answer_span_via_type(start_idx=child_start, tree_node=child,
                                               sparse_graph=sparse_graph,
                                               valuable_parents=valuable_parents)
            child_start += len(child.leaves())
        else:
            child_start += 1
    assert child_start == start_idx + len(span_tokens), '{}: {}: {}'.format(
        child_start, len(span_tokens), tree_node
    )
    if add_flag is True:
        valuable_parents.pop()
    return spans


def find_span_pos(sent: str):
    """Find span positions. Those spans can be NP, NNP, VP, SBAR, ADJP
        and cannot be in stop_words"""
    # build a sparse graph linking related spans. If one span is selected, other spans linked will be deleted from the candidate set
    sparse_graph = {} # 
    sentence = ' '.join(sent.split()[:65])
    s = bene_parser.parse(sentence)
    return (s, find_answer_span_via_type(start_idx=0, tree_node=s[0],
                                         sparse_graph=sparse_graph), sparse_graph)


def spanize(sent: nltk.Tree, raw_spans: List[Tuple[str, str, Tuple[int, int]]],
            sparse_graph: Mapping[Tuple[int, int], List[Tuple[int, int]]],
            paper_id: str, sec_idx: int, para_idx: int,
            sent_idx: int) -> Tuple[List[List[Token]], List[bool],
                                    List[Tuple[str, str, Tuple[int, int], int]]]:
    """Determine the final spans based on T5 reconstruction loss. The harder the span
        is to be reconstructed, the more informative it is.

    Args:
        sent (nltk.Tree): _description_
        raw_spans (List[Tuple[str, str, List[int]]]): _description_
    """
    tokens = sent.leaves()
    raw_spans_with_score = []
    final_spans = []
    bsz = 12
    if paper_id == '1910.14497':
        bsz = 4
    raw_span_len = len(raw_spans)
    for raw_span_s_idx in range(0, raw_span_len, bsz):
        raw_span_e_idx = min(raw_span_s_idx + bsz, raw_span_len)
        masked_txts = []
        label_txts = []
        real_bsz = raw_span_e_idx - raw_span_s_idx
        for span_idx in range(raw_span_s_idx, raw_span_e_idx):
            raw_span = raw_spans[span_idx]
            s_idx = raw_span[2][0]
            e_idx = raw_span[2][1]
            masked_txt = ' '.join(tokens[: s_idx]) + ' <extra_id_0> ' + ' '.join(tokens[e_idx:])
            label_txt = '<extra_id_0> ' + raw_span[0] + ' <extra_id_1>'
            masked_txts.append(masked_txt)
            label_txts.append(label_txt)
        input_ids = t5_tokenizer(masked_txts, return_tensors="pt", padding=True).input_ids.to(device)
        labels = t5_tokenizer(label_txts, return_tensors="pt", padding=True).input_ids.to(device)
        res = t5_model(input_ids=input_ids, labels=labels)
        lm_logits = res.logits
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss_vec = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).view(real_bsz, -1).mean(-1).tolist()
        assert len(loss_vec) == raw_span_e_idx - raw_span_s_idx
        raw_spans_with_score.extend(raw_spans[span_idx] + (loss_vec[span_idx - raw_span_s_idx],)
                                    for span_idx in range(raw_span_s_idx, raw_span_e_idx))
    # the greater the loss, the hard for the model to generate the span, the more valuable the information is
    raw_spans_with_score = sorted(raw_spans_with_score, key=lambda x: -x[-1])
    # collect all position ranges: [s_idx, e_idx)
    candidate_spans = set(raw_span[2] for raw_span in raw_spans_with_score)
    # select candidate spans while removing all its parent spans or child spans
    while len(candidate_spans) > 0:
        for raw_span in raw_spans_with_score:
            cur_pos_range = raw_span[2]
            if cur_pos_range not in candidate_spans:
                continue
            candidate_spans.remove(cur_pos_range)
            for related_range in sparse_graph[cur_pos_range]:
                if related_range in candidate_spans:
                    candidate_spans.remove(related_range)
            final_spans.append(raw_span)

    # sort final spans according to their positions in the sentence
    final_spans = sorted(final_spans, key=lambda x: x[-2])
    spanized_seq_str = []
    is_trivial_span = []
    s_idx = 0
    for final_span in final_spans:
        pos_range = final_span[-2]
        cur_s_idx = pos_range[0]
        cur_e_idx = pos_range[1]
        if s_idx < cur_s_idx:
            # a trivial span is a span in the middle of two final spans 
            # or in the leftmost/rightmost while not in a final span)
            spanized_seq_str.append(' '.join(tokens[s_idx: cur_s_idx]))
            is_trivial_span.append(True)
        else:
            assert s_idx == cur_s_idx
        spanized_seq_str.append(' '.join(tokens[cur_s_idx: cur_e_idx]))
        is_trivial_span.append(False)
        s_idx = cur_e_idx
    if s_idx < len(tokens):
        spanized_seq_str.append(' '.join(tokens[s_idx:]))
        is_trivial_span.append(True)

    spanized_seq = [] # List[List[Token]]
    final_span_idx = 0
    # tokenize each str of a span and save them into spanized_seq
    for span_str, is_trivial in zip(spanized_seq_str, is_trivial_span):
        subtoken_strs = tokenizer.tokenize(span_str)
        span_tokens = []
        if is_trivial:
            final_span_id = -1
        else:
            final_span_id = final_span_idx
            final_span_idx += 1
        for subtoken_str in subtoken_strs:
            token_id = tokenizer.convert_tokens_to_ids(subtoken_str)
            span_tokens.append(
                Token(s=subtoken_str, token_id=token_id, paper_id=paper_id, sec_idx=sec_idx,
                      para_idx=para_idx, sent_id=sent_idx, final_span_id=final_span_id)
            )
        assert len(span_tokens) > 0
        spanized_seq.append(span_tokens)
    assert len(spanized_seq) == len(is_trivial_span)
    return (spanized_seq, is_trivial_span, raw_spans_with_score)


def sample_span(spans, has_global_edges: List[bool] = []):
    """sample a subset of spans

    Args:
        spans (_type_): _description_
    """
    TOP = 24 + 8
    TOP_RANDOM = 24
    TOP_GLOBAL = 8
    sampled_spans = []
    cnt_multi_spans = sum(len(span) > 1 for span in spans)
    candidate_multi_spans = []
    candidate_single_spans = []
    for span in spans:
        if len(span) == 1:
            candidate_single_spans.append(span)
        else:
            candidate_multi_spans.append(span)
    candidate_single_spans = sorted(candidate_single_spans,
                                    key=lambda x: -len(x[0][0].split()))[: 3 * len(candidate_multi_spans)]

    candidate_spans = candidate_single_spans + candidate_multi_spans
    random.shuffle(candidate_spans)
    candidate_spans = candidate_spans[: TOP_RANDOM]
    candidate_global_spans = []
    for span, has_global_edge in zip(spans, has_global_edges):
        if has_global_edge and span not in candidate_spans:
            candidate_global_spans.append(span)
    random.shuffle(candidate_global_spans)
    is_global = [False] * min(TOP_RANDOM, len(candidate_spans)) + [True] * min(TOP_GLOBAL, len(candidate_global_spans))
    candidate_spans += candidate_global_spans[: TOP_GLOBAL]
    return candidate_spans, is_global

def main():
    input_file = '../data/qasper-train-v0.3.json'
    # input_file = '/data1/home/nyx/qa/longuqa/ulqa-debug.json'
    # output_dir = './ulqa_data'
    # output_dir = './ulqa_data_debug_v2'
    # output_dir = './ulqa_data_debug_v3'
    output_dir = './ulqa_data_v3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    existing_output_files = glob.glob(os.path.join(output_dir, '*.json'))
    existing_paper_ids = set()
    for file in existing_output_files:
        basename = ntpath.basename(file).replace('.json', '')
        if len(basename.split('.')) == 2:
            first_half = basename.split('.')[0]
            second_half = basename.split('.')[1]
            if (first_half.isdigit() and len(first_half) == 4
                and second_half.isdigit() and len(second_half) == 5):
                existing_paper_ids.add(basename)

    print('Find [{}] existing output files.'.format(len(existing_paper_ids)))
    global stop_spans
    data_range = os.environ['DATA_RANGE']
    data_range = [int(data_range.split('-')[0]), int(data_range.split('-')[1])]
    paper_id_to_paras, paper_ids = convert_qasper_context(input_file, stop_spans,
                                                          existing_paper_ids,
                                                          data_range[0])

    paper_to_be_processed = {}
    paper_id_to_be_processed = []
    paper_cur_cnt = 0
    for paper_id, paper in paper_id_to_paras.items():
        paper_cur_cnt += 1
        if paper_id in existing_paper_ids:# or paper_cur_cnt < data_range[1] - data_range[0]:
            continue
        if paper_cur_cnt > data_range[1] - data_range[0]:
            break
        paper_to_be_processed[paper_id] = paper
        paper_id_to_be_processed.append(paper_id)
    paper_id_to_paras = paper_to_be_processed
    paper_ids = paper_id_to_be_processed
    print('[{}] papers need to be processed!'.format(len(paper_ids)))
    
    output_qas = []
    output_is_global = []
    TOP = 10000
    MAX_LEN = 10000
    for paper_id, paras in tqdm(paper_id_to_paras.items(), total=len(paper_id_to_paras), desc='process papers...'):
        Final_raw_spans = []
        Final_is_trivial_span = []
        Final_is_global_span = []
        full_text_tokens = []
        if len(paras) == 0:
            continue
        output_qas = []
        paper_sentnized = [] # List[List[str]], Section[Paragraph[Sentence]]
        cur_len = 0
        for sec_idx, sec in enumerate(tqdm(paras, desc='paper [{}] spanized...'.format(paper_id))):
            section_name = sec['section_name']
            paras = [section_name] + sec['paragraphs']
            paras_sentnized = [sent_tokenize(p) for p in paras]
            cur_len += sum(len(sent.split()) for p in paras_sentnized for sent in p)
            if cur_len >= MAX_LEN:
                break
            # find potential spans via constituent parsing
            span_positions = [find_span_pos(sent) + (paper_id, sec_idx, para_idx, sent_idx)
                              for para_idx, para in enumerate(paras_sentnized)
                                for sent_idx, sent in enumerate(para)]
            # determine the final spans via slot filling
            # sent: (Tree, List[spans[str, label, s_idx, e_idx]], 
            #        sparse_graph, paper_id, sec_idx, para_idx, sent_idx)
            # paper_spans: A list, each element is a tuple, representing a sentence.
            # For each sentence tuple, it aims to store spans in the sentence, where each span
            # is tokenized by a pretrained tokenizer.
            # Three elements are in a sentence tuple:
            # 1. spanized_seq, List[List[Token]]: each element in the list is a span, 
            #   while each element in a
            #   span (also a list) is a Token object, which stores information of a specific subtoken
            #   in the span
            # 2. is_trivial_span, List[bool]: It has the same length with the list in 1. For each element,
            #   if it is False, that means that the corresponding span in 1. is a span selected for answer
            #   generation subsequently. If it is True, then the span is not selected.
            # 3. raw_spans_with_score: List[...]: Similar to the return list in 'find_span_pos',
            #   each element is attached to a specific score, which represents the importance of that span.
            paper_spans = [spanize(*sent) for sent in span_positions]
            raw_spans = []
            # add <s> tokens at the beginning of the sequence
            # raw_spans = [paper_span[0] for paper_span in paper_spans]
            # Flatten those spans. It means that sentence boundaries are diminished.
            cur_para_idx = None
            for paper_span in paper_spans:
                if cur_para_idx is None or cur_para_idx < paper_span[0][0][0].para_idx:
                    cur_para_idx = paper_span[0][0][0].para_idx
                    raw_spans.append([[Token(s='</s>', token_id=tokenizer.convert_tokens_to_ids('</s>'),
                                       paper_id=paper_id, sec_idx=-1, para_idx=cur_para_idx,
                                       sent_id=-1, final_span_id=-1)]])
                    Final_raw_spans += raw_spans[-1]
                    Final_is_trivial_span += [True]
                    Final_is_global_span += [True]

                raw_spans.append(paper_span[0])
                Final_raw_spans += raw_spans[-1]
                Final_is_trivial_span += paper_span[1]
                Final_is_global_span += [False] * len(paper_span[1])
            # full_text_tokens: flattened Token list
            full_text_tokens += [token for sent_span_tokens in raw_spans
                                    for span_tokens in sent_span_tokens
                                        for token in span_tokens]
            paper_sentnized.append(paras_sentnized)
        assert len(Final_raw_spans) == len(Final_is_trivial_span), '{}: {}'.format(
            len(Final_raw_spans), len(Final_is_trivial_span)
        )
        assert len(Final_is_global_span) == len(Final_is_trivial_span), '{}: {}'.format(
            len(Final_is_global_span), len(Final_is_trivial_span)
        )
        inputs = torch.tensor(tokenizer.convert_tokens_to_ids([token.s for token in full_text_tokens]), dtype=torch.long,
                              device=device1)
        seq_len = inputs.size(0)
        print('seq_len: [{}]'.format(seq_len))
        global_attention_mask = torch.tensor([token.s == '<s>' or token.s == '</s>' for token in full_text_tokens],
                                             dtype=torch.bool, device=device1)
        global_token_num = torch.sum(global_attention_mask).cpu().item()
        print('global_token_num: [{}]'.format(global_token_num))
        inputs.unsqueeze_(0)
        global_attention_mask.unsqueeze_(0)
        outputs = encoder(input_ids=inputs, global_attention_mask=global_attention_mask, output_attentions=True)
        global_attention_r2l = outputs[-1] # [batch_size, max_seq_len (a number greater that seq len), global_token_num]
        global_attention_r2l = tuple(item.cpu() for item in global_attention_r2l)
        cleaned_global_attention = []
        # remove paddings in the seq dim
        for ga in global_attention_r2l:
            _ga = ga[:, :, :seq_len, :]
            cleaned_global_attention.append(_ga)
        global_attention_r2l = tuple(cleaned_global_attention)

        full_text_attention = outputs[-2]  # [batch_size, max_seq_len]
        full_text_attention = tuple(item.cpu() for item in full_text_attention)
        for item in full_text_attention:
            assert not item.is_cuda
        # separate global attention from local attention
        global_attention_l2r = []
        real_full_text_attention = []
        for fta in full_text_attention:
            _global_attention_l2r = fta[:, :, :, :global_token_num]
            _local_attention = fta[:, :, :, global_token_num:]
            global_attention_l2r.append(_global_attention_l2r)
            real_full_text_attention.append(_local_attention)

        global_attention_l2r = tuple(global_attention_l2r)
        full_text_attention = tuple(real_full_text_attention)
        # check valid
        for l2r, r2l in zip(global_attention_l2r, global_attention_r2l):
            assert l2r.size() == r2l.size(), '{}: {}'.format(
                l2r.size(), r2l.size()
            )
            assert l2r.size(0) == 1
            assert l2r.size(1) == 12
            assert l2r.size(2) == seq_len
            assert l2r.size(3) == global_token_num
        for local_attention in full_text_attention:
            assert local_attention.size(0) == 1
            assert local_attention.size(1) == 12
            assert local_attention.size(2) == seq_len
            assert local_attention.size(3) == attention_window_size + 1
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])
        seq = tokenizer.convert_tokens_to_string(tokens)
        logging.info('full_text_attention: ')
        logging.info((len(full_text_attention),) + full_text_attention[0].size())
        head_nums = 12
        layer_nums = 6
        final_spans = []
        print('original span nums: [{}]'.format(len(Final_raw_spans)))
        trivial_span_ids = {span_idx for span_idx, label in enumerate(Final_is_trivial_span)
                                       if label is True}
        global_span_ids = [span_idx for span_idx, label in enumerate(Final_is_global_span)
                                        if label is True]
        print('remaining span nums: [{}]'.format(len(Final_is_trivial_span) - len(trivial_span_ids)))
        assert isinstance(trivial_span_ids, set)
        final_spans, final_has_global_edges = span_collect_with_multi_processes(paper_id, layer_nums, head_nums, Final_raw_spans,
                                                                                trivial_span_ids,
                                                                                full_text_attention, attention_window_size,
                                                                                tokenizer,
                                                                                global_span_ids,
                                                                                global_attention_l2r,
                                                                                global_attention_r2l)

        logging.info('final_spans: {}'.format(final_spans))
        logging.info('final_has_global_edges: {}'.format(final_has_global_edges))
        print('{}/{} spans are with global edges'.format(sum(final_has_global_edges),
                                                         len(final_has_global_edges)))
        assert len(final_spans) == len(final_has_global_edges), '{}: {}'.format(
            len(final_spans), len(final_has_global_edges)
        )
        filtered_spans, is_global = sample_span(final_spans, final_has_global_edges)
        assert len(filtered_spans) == len(is_global), '{}: {}'.format(
            len(filtered_spans), len(is_global)
        )
        if os.environ['AGGREGATOR'] == 't5':
            print('Using T5 Model for aggregating answers')
            aggregated_spans = answer_aggregator_via_t5(filtered_spans)
        else:
            print('Using BART Model for aggregating answers')
            aggregated_spans = answer_aggregator_via_bart(filtered_spans)
        assert len(filtered_spans) == len(is_global), '{}: {}'.format(
            len(filtered_spans), len(is_global)
        )
        logging.info('aggregated_spans: {}'.format(aggregated_spans))
        ppl_filtered_spans = aggregated_spans
        direct_to_long_question(sents=paper_sentnized, spans=ppl_filtered_spans)
        logging.info('Long qa pairs for each span: {}'.format([item.long_qas for item in ppl_filtered_spans]))
        long_qas = []
        for item in ppl_filtered_spans:
            long_qas += item.long_qas[0]
        long_qas, is_global = heuristic_filter(long_qas, is_global)
        print('Filtered Long qa pair number: [{}]'.format(len(long_qas)))
        logging.info('Filtered Long qa pairs for each span: {}'.format([item.long_qas for item in ppl_filtered_spans]))

        output_qas += ppl_filtered_spans
        output_is_global += is_global

        write_to_file(input_file, paper_id, paper_id_to_paras, paper_sentnized, output_qas, os.path.join(output_dir, paper_id + '.json'),
                      output_is_global)

if __name__ == '__main__':
    main()
