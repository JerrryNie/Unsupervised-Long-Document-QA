from typing import List, Dict, Set 
import torch
from transformers import AutoTokenizer
from modules import Token, Span
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from scipy.stats import gmean
import os

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.array(x) / np.sum(x, axis=0)

case_study_snippets = [
    "The main contributions",
    "a single-layer forward recurrent neural network",
    "Long Short-Term Memory"
]

class Node:
    """A Node contains a continuous sequence of subtokens, i.e. a span
    """
    def __init__(self, idx: int, span_subtokens: List[Token], start_pos: int, half_window_size: int,
                 total_len: int, tokenizer: AutoTokenizer):
        self.idx = idx
        assert isinstance(span_subtokens, list)
        assert isinstance(span_subtokens[0], Token), '{}; {}'.format(span_subtokens[0], type(span_subtokens[0]))
        self.subtokens = span_subtokens
        self.span = tokenizer.convert_tokens_to_string([subtoken.s for subtoken in span_subtokens])
        self.pos = [i + start_pos for i in range(len(span_subtokens))]
        self.start_pos = start_pos
        self.lmost_reached_subtoken_idx = max(0, self.pos[0] - half_window_size)
        self.rmost_reached_subtoken_idx = min(total_len - 1, self.pos[-1] + half_window_size)


class SubtokenEdge:
    def __init__(self, src_idx, tgt_idx, value) -> None:
        self.src_idx = src_idx
        self.tgt_idx = tgt_idx
        self.value = value


class Edge:
    def __init__(self, src_idx, tgt_idx, value=None):
        self.src_idx = src_idx
        self.tgt_idx = tgt_idx
        self.value = value
        self.bi_v = None


class Graph:
    def __init__(self, seq: List[List[Token]], trivial_span_ids: Set[int],half_window_size: int, attention: torch.Tensor,
                 tokenizer: AutoTokenizer, global_span_ids: List[int] = [],
                 ga_l2r: torch.Tensor = None, ga_r2l: torch.Tensor = None,
                 layer_idx: int = -1, head_idx: int = -1, pooling: str = 'max_pooling'):
        self.half_window_size = half_window_size
        self.tokenizer = tokenizer
        self.nodes = []
        self.edges = {}
        self.global_edges = {}
        self.edges_src = []
        self.edges_tgt = []
        self.global_span_to_local_span = {} # Dict[int, List[int]]
        self.local_span_to_global_span = {} # Dict[int, int]
        self.subtoken_edges = {}
        self.subtoken_id_to_node_id = {}
        self.seq = seq
        self.attention = attention # [sequence_len, attention_window_size]
        self.max_value = -1e-5
        self.min_value = 1e5
        self.attention_scores = []
        self.trivial_nodes = trivial_span_ids
        self.global_span_ids = global_span_ids
        self.ga_l2r = ga_l2r.tolist() if ga_l2r is not None else None # [seq_len, global_token_num]
        self.ga_r2l = ga_r2l.tolist() if ga_r2l is not None else None # [seq_len, global_token_num]
        self.global_to_global_attention = {} # Dict[int, List[Tuple[int, float]]]
        self.global_to_local_attention = {}
        self.valuable_global_edges = set()
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.pooling = pooling

    def parse_seq(self):
        start_pos = 0
        total_len = sum([len(s) for s in self.seq])
        for node_idx, span_subtokens in enumerate(self.seq):
            self.nodes.append(Node(idx=node_idx, span_subtokens=span_subtokens,                 
                                   start_pos=start_pos,
                                   half_window_size=self.half_window_size, total_len=total_len,
                                   tokenizer=self.tokenizer))
            for idx in range(len(span_subtokens)):
                subtoken_idx = start_pos + idx
                self.subtoken_id_to_node_id[subtoken_idx] = node_idx
                if node_idx not in self.trivial_nodes:
                    subtoken_attention = self.attention[subtoken_idx].tolist()
                    for attention_idx, attention_value in enumerate(subtoken_attention):
                        relative_pos = attention_idx - self.half_window_size
                        tgt_subtoken_idx = relative_pos + subtoken_idx
                        if tgt_subtoken_idx >= 0:
                            self.subtoken_edges[(subtoken_idx, tgt_subtoken_idx)] = SubtokenEdge(
                                src_idx=subtoken_idx, tgt_idx=tgt_subtoken_idx, value=attention_value
                            )
            start_pos += len(span_subtokens)
        self.edges_src = [[] for _ in self.nodes]
        self.edges_tgt = [[] for _ in self.nodes]
        if self.global_span_ids is not None and len(self.global_span_ids) > 0:
            cur_global_span = None
            cur_para_idx = None
            for g_idx, global_node_idx in enumerate(self.global_span_ids):
                if global_node_idx not in self.global_to_local_attention:
                    self.global_to_local_attention[global_node_idx] = []
                global_node = self.nodes[global_node_idx]
                cur_global_span = global_node.subtokens
                assert len(cur_global_span) == 1 and cur_global_span[0].s == '</s>'
                self.global_span_to_local_span[global_node_idx] = []
                cur_para_idx = cur_global_span[0].para_idx
                cur_global_pos = global_node.start_pos
                s_span_idx = global_node_idx + 1
                e_span_idx = self.global_span_ids[g_idx + 1] if g_idx + 1 < len(self.global_span_ids) else len(self.nodes)
                cur_global_to_local_attention_scores = []
                for span_idx in range(s_span_idx, e_span_idx):
                    cur_span = self.nodes[span_idx]
                    cur_span_pos = cur_span.start_pos
                    if span_idx not in self.trivial_nodes:
                        for subtoken_idx, subtoken in enumerate(cur_span.subtokens):
                            cur_pos = cur_span_pos + subtoken_idx
                            cur_edge_value_r2l = self.ga_r2l[cur_pos][g_idx]
                            cur_global_to_local_attention_scores.append(cur_edge_value_r2l)
                
                if cur_global_to_local_attention_scores:
                    cur_global_to_local_attention_scores = softmax(cur_global_to_local_attention_scores).tolist()

                subtoken_pointer = 0
                for span_idx in range(s_span_idx, e_span_idx):
                    if span_idx in self.trivial_nodes:
                        continue
                    self.global_span_to_local_span[global_node_idx].append(span_idx)
                    assert span_idx not in self.local_span_to_global_span
                    self.local_span_to_global_span[span_idx] = global_node_idx
                    cur_span = self.nodes[span_idx]
                    assert cur_span.subtokens[0].para_idx == cur_para_idx
                    edge_value_l2r = -1 if self.pooling != 'min_pooling' else 100
                    edge_value_l2r_list = []
                    edge_value_r2l = -1 if self.pooling != 'min_pooling' else 100
                    edge_value_r2l_list = []
                    cur_span_pos = cur_span.start_pos
                    for subtoken_idx, subtoken in enumerate(cur_span.subtokens):
                        cur_pos = cur_span_pos + subtoken_idx
                        cur_edge_value_l2r = self.ga_l2r[cur_pos][g_idx]
                        cur_edge_value_r2l = cur_global_to_local_attention_scores[subtoken_pointer]
                        subtoken_pointer += 1
                        if self.pooling == 'max_pooling' or self.pooling == 'mean_pooling':
                            edge_value_l2r = max(edge_value_l2r, cur_edge_value_l2r)
                            edge_value_r2l = max(edge_value_r2l, cur_edge_value_r2l)
                        elif self.pooling == 'min_pooling':
                            edge_value_l2r = min(edge_value_l2r, cur_edge_value_l2r)
                            edge_value_r2l = min(edge_value_r2l, cur_edge_value_r2l)
                        edge_value_l2r_list.append(cur_edge_value_l2r)
                        edge_value_r2l_list.append(cur_edge_value_r2l)
                    if self.pooling == 'mean_pooling':
                        edge_value_l2r = sum(edge_value_l2r_list) / len(edge_value_l2r_list)
                        edge_value_r2l = sum(edge_value_r2l_list) / len(edge_value_r2l_list)
                    self.global_edges[(span_idx, global_node_idx)] = Edge(src_idx=span_idx, tgt_idx=global_node_idx, value=edge_value_l2r)
                    self.global_edges[(global_node_idx, span_idx)] = Edge(src_idx=global_node_idx, tgt_idx=span_idx, value=edge_value_r2l)
                    self.global_to_local_attention[global_node_idx].append((span_idx, edge_value_r2l))

            global_to_global_attention = []
            for idx in self.global_span_ids:
                global_to_global_attention.append(self.ga_r2l[idx])
            assert len(global_to_global_attention) == len(global_to_global_attention[0])
            for s_g_idx, s_global_node_idx in enumerate(self.global_span_ids):
                attention_vector = []
                for e_g_idx, e_global_node_idx in enumerate(self.global_span_ids):
                    if s_g_idx == e_g_idx:
                        continue
                    s_g_node = self.nodes[s_global_node_idx]
                    e_g_node = self.nodes[e_global_node_idx]
                    edge_value = global_to_global_attention[e_g_idx][s_g_idx]
                    attention_vector.append(edge_value)
                attention_vector = softmax(attention_vector)
                attention_pointer = 0
                for e_g_idx, e_global_node_idx in enumerate(self.global_span_ids):
                    if s_g_idx == e_g_idx:
                        continue
                    edge_value = attention_vector[attention_pointer]
                    attention_pointer += 1
                    self.global_edges[(s_global_node_idx, e_global_node_idx)] = Edge(src_idx=s_global_node_idx, tgt_idx=e_global_node_idx, value=edge_value)
                    if s_global_node_idx not in self.global_to_global_attention:
                        self.global_to_global_attention[s_global_node_idx] = []
                    self.global_to_global_attention[s_global_node_idx].append((e_global_node_idx, edge_value))

    def build_edges(self):
        for node_idx, node in enumerate(self.nodes):
            if node.idx in self.trivial_nodes:
                continue
            l_most_subtoken = node.lmost_reached_subtoken_idx
            r_most_subtoken = node.rmost_reached_subtoken_idx
            l_most_node_idx = self.subtoken_id_to_node_id[l_most_subtoken]
            r_most_node_idx = self.subtoken_id_to_node_id[r_most_subtoken]
            for tgt_node_idx in range(l_most_node_idx, r_most_node_idx + 1):
                if tgt_node_idx != node.idx and tgt_node_idx not in self.trivial_nodes:
                    self.edges[(node.idx, tgt_node_idx)] = Edge(src_idx=node.idx, tgt_idx=tgt_node_idx)
                    self.edges_src[node.idx].append(tgt_node_idx)
                    self.edges_tgt[tgt_node_idx].append(node.idx)

    def calculate_edge_values(self):
        for src_node_idx, tgt_nodes in enumerate(self.edges_src):
            src_subtokens = self.nodes[src_node_idx].pos
            txt_src_subtokens = self.nodes[src_node_idx].subtokens
            for tgt_node_idx in tgt_nodes:
                tgt_subtokens = self.nodes[tgt_node_idx].pos
                txt_tgt_subtokens = self.nodes[tgt_node_idx].subtokens
                assert src_node_idx not in self.trivial_nodes
                assert tgt_node_idx not in self.trivial_nodes
                max_edge_value = -1e10
                min_edge_value = 1000
                edge_value_list = []
                for src_idx, src_subtoken_idx in enumerate(src_subtokens):
                    for tgt_idx, tgt_subtoken_idx in enumerate(tgt_subtokens):
                        if (src_subtoken_idx, tgt_subtoken_idx) in self.subtoken_edges:
                            cur_edge_value = self.subtoken_edges[(src_subtoken_idx, tgt_subtoken_idx)].value
                            if self.head_idx == 9 and self.layer_idx == 5 and src_node_idx == 711 and tgt_node_idx == 713:
                                print('src_subtoken: {}, tgt_subtoken: {}, ({}, {})={}'.format(
                                    txt_src_subtokens[src_idx].s, txt_tgt_subtokens[tgt_idx].s,
                                    src_idx, tgt_idx, cur_edge_value
                                ))
                            max_edge_value = max(max_edge_value, cur_edge_value)
                            min_edge_value = min(min_edge_value, cur_edge_value)
                            edge_value_list.append(cur_edge_value)
                assert (src_node_idx, tgt_node_idx) in self.edges, '({}, {})'.format(
                    src_node_idx, tgt_node_idx
                )
                if self.pooling == 'max_pooling':
                    update_value = max_edge_value
                elif self.pooling == 'min_pooling':
                    update_value = min_edge_value
                elif self.pooling == 'mean_pooling':
                    update_value = sum(edge_value_list) / len(edge_value_list)
                else:
                    raise Exception('')
                self.edges[(src_node_idx, tgt_node_idx)].value = update_value

    def update_edge_values_via_global_attention(self, local_to_global_threshold=0.01, top_k=3):
        for node_idx, node in enumerate(self.nodes):
            if node_idx in self.trivial_nodes:
                continue
            global_node_idx = self.local_span_to_global_span[node_idx]
            l2g_edge_value = self.global_edges[(node_idx, global_node_idx)].value
            if l2g_edge_value < local_to_global_threshold:
                continue
            tgt_global_nodes = self.global_to_global_attention[global_node_idx]
            selected_global_nodes = sorted(tgt_global_nodes, key=lambda x: -x[1])[:top_k]
            for tgt_global_node in selected_global_nodes:
                g2g_edge_value = self.global_edges[(global_node_idx, tgt_global_node[0])].value
                assert g2g_edge_value == tgt_global_node[1]
                tgt_local_nodes = self.global_to_local_attention[tgt_global_node[0]]
                selected_local_nodes = sorted(tgt_local_nodes, key=lambda x: -x[1])[:top_k]
                for tgt_local_node in selected_local_nodes:
                    g2l_edge_value = self.global_edges[(tgt_global_node[0], tgt_local_node[0])].value
                    assert g2l_edge_value == tgt_local_node[1]
                    geo_mean = gmean([l2g_edge_value, g2g_edge_value, g2l_edge_value])
                    geo_mean = float(geo_mean)
                    if (node_idx, tgt_local_node[0]) not in self.edges or self.edges[(node_idx, tgt_local_node[0])].value < geo_mean:
                        self.valuable_global_edges.add((node_idx, tgt_local_node[0]))
                        if node_idx == 264 and tgt_local_node[0] == 711 and geo_mean >= 0.45:
                            print('layer [{}], head [{}], geo_mean from 264 to 711 is: [{}]'.format(self.layer_idx, self.head_idx, geo_mean))
                        self.edges[(node_idx, tgt_local_node[0])] = Edge(src_idx=node_idx, tgt_idx=tgt_local_node[0], value=geo_mean)
                        self.edges_src[node_idx].append(tgt_local_node[0])

    def span_collector_via_dfs(self, threshold):
        self.walking_threshold = threshold
        self.left_to_right_candidates = [] # List of List
        self.right_to_left_candidates = []
        self.record_visited = set()
        self.record_visited_reverse = set()
        for idx in range(len(self.nodes)):
            self.visited = set()
            self.visited_list = []
            if idx in self.trivial_nodes:
                continue
            if idx not in self.record_visited:
                self.dfs(src_node_idx=idx, reverse=False)
            if len(self.nodes) - 1 - idx not in self.record_visited_reverse:
                self.dfs(src_node_idx=len(self.nodes) - 1 - idx, reverse=True)
    
    def dfs(self, src_node_idx: int, reverse: bool):
        """DFS through nodes

        Args:
            src_node (int): start_node_idx
            reverse (bool): false: left-to-right, true: right-to-left
        """
        if src_node_idx in self.trivial_nodes:
            return None
        self.visited.add(src_node_idx)
        if reverse is False:
            self.record_visited.add(src_node_idx)
        else:
            self.record_visited_reverse.add(src_node_idx)
        self.visited_list.append(src_node_idx)
        search_tag = False
        for tgt_node_idx in self.edges_src[src_node_idx]:
            if tgt_node_idx in self.trivial_nodes:
                continue
            if self.edges[(src_node_idx, tgt_node_idx)].value >= self.walking_threshold:
                if reverse is False and tgt_node_idx > src_node_idx and tgt_node_idx not in self.record_visited:
                    search_tag = True
                    self.dfs(src_node_idx=tgt_node_idx, reverse=reverse)
                elif reverse is True and tgt_node_idx < src_node_idx and tgt_node_idx not in self.record_visited_reverse:
                    search_tag = True
                    self.dfs(src_node_idx=tgt_node_idx, reverse=reverse)

        if search_tag is False:
            if reverse is False:
                self.left_to_right_candidates.append(deepcopy(self.visited_list))
            else:
                self.right_to_left_candidates.append(deepcopy(self.visited_list))
        self.visited.remove(src_node_idx)
        self.visited_list.pop()

    def get_graph_clustered_spans(self):
        l2r = []
        for item in self.left_to_right_candidates:
            candidate_path = [(idx, self.nodes[idx].span, self.nodes[idx].subtokens) for idx in item]
            l2r.append(candidate_path)
        r2l = []
        r2l_edges = set()
        for item in self.right_to_left_candidates:
            candidate_path = [(idx, self.nodes[idx].span, self.nodes[idx].subtokens) for idx in item]
            r2l.append(candidate_path)
        for item in r2l:
            for s_idx in range(len(item) - 1):
                e_idx = s_idx + 1
                s_edge = item[s_idx][0]
                e_edge = item[e_idx][0]
                r2l_edges.add((s_edge, e_edge))
        l2r_edges = set()
        for item in l2r:
            for s_idx in range(len(item) - 1):
                e_idx = s_idx + 1
                s_edge = item[s_idx][0]
                e_edge = item[e_idx][0]
                if (e_edge, s_edge) in r2l_edges:
                    l2r_edges.add((s_edge, e_edge))
        double_checked_l2r = []
        
        return {'l2r': l2r, 'r2l': r2l, 'double_checked_l2r': double_checked_l2r}


if __name__ == '__init__':
    model_name = 'allenai/led-base-16384' # 'google/long-t5-tglobal-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    graph = Graph()
