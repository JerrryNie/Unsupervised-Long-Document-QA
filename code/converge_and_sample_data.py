"""Converge all the data files in ./ulqa_data 
    into a single file ./ulqa_qasper_train.json and sample some data to initialize a QA model"""
import glob
import json
from copy import deepcopy
import random
from tqdm import tqdm
import os
import sys
from ntpath import basename
random.seed(42)
# sample instances for training
SAMPLE_NUM = 10000
VERSION = 2
input_train_file = 'qasper-train-v0.3.json'
output_dir = sys.argv[1] # [./ulqa_data_v2, ./ulqa_data_v3]
output_file = os.path.join(output_dir, 'ulqa_qasper_train.json')
output_sampled_file = os.path.join(output_dir, 'ulqa_qasper_train_{}.json'.format(SAMPLE_NUM))
output_sampled_revised_file = os.path.join(output_dir, 'ulqa_qasper_train_{}_revised.json'.format(SAMPLE_NUM))
qid_to_paper_id = {}
generated_qas_files = glob.glob(os.path.join(output_dir, '*.json'))
generated_paper_id_to_qas = {}
for qas_file in generated_qas_files:
    if basename(qas_file).startswith('ulqa'):
        continue
    with open(qas_file, 'r', encoding='utf-8') as f:
        qas = json.load(f)
    assert len(qas) == 1, '{}'.format(qas_file)
    for key, paper in qas.items():
        generated_paper_id_to_qas[key] = paper
        for qa in paper['qas']:
            qid = qa['question_id']
            assert qid not in qid_to_paper_id
            qid_to_paper_id[qid] = key

with open(input_train_file, 'r', encoding='utf-8') as f:
    qasper = json.load(f)

_qasper = deepcopy(qasper)
for key, paper in _qasper.items():
    if len(paper['full_text']) == 0:
        paper['qas'] = []
        continue
    if key not in generated_paper_id_to_qas:
        print('Warning: 0 QA pairs generated for paper [{}]'.format(key))
        paper['qas'] = []
        continue
    paper['qas'] = generated_paper_id_to_qas[key]['qas']

if not os.path.exists(output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(_qasper, f, indent=2, ensure_ascii=False)
else:
    print('Warning: [{}] exists, skip it...'.format(output_file))

qids = list(qid_to_paper_id.keys())
print('Total qa pairs: [{}]'.format(len(qids)))
random.shuffle(qids)

remaining_qids = qids[: SAMPLE_NUM]
remaining_paper_ids = {qid_to_paper_id[qid] for qid in remaining_qids}

_qasper = deepcopy(qasper)

for key, paper in tqdm(_qasper.items(), total=len(_qasper), desc='sampling...'):
    if len(paper['full_text']) == 0:
        paper['qas'] = []
        continue
    if key not in remaining_paper_ids:
        paper['qas'] = []
        continue
    assert key in remaining_paper_ids, '{}'.format(key)
    candidate_qas = generated_paper_id_to_qas[key]['qas']
    qas = [qa for qa in candidate_qas if qa['question_id'] in remaining_qids]
    paper['qas'] = qas

if not os.path.exists(output_sampled_file):
    with open(output_sampled_file, 'w', encoding='utf-8') as f:
        json.dump(_qasper, f, indent=2, ensure_ascii=False)
else:
    print('Warning: [{}] exists, skip it...'.format(output_sampled_file))

cnt_qas = 0
for key, paper in tqdm(_qasper.items(), total=len(_qasper), desc='sampling...'):
    if len(paper['full_text']) == 0:
        paper['qas'] = []
        continue
    qas = [qa for qa in paper['qas'] if not (qa['answers'][0]['answer']['free_form_answer'].startswith('$')\
        and qa['answers'][0]['answer']['free_form_answer'].endswith('$'))]
    cnt_qas += len(qas)
    paper['qas'] = qas

print(cnt_qas)
if not os.path.exists(output_sampled_revised_file):
    with open(output_sampled_revised_file, 'w', encoding='utf-8') as f:
        json.dump(_qasper, f, indent=2, ensure_ascii=False)
else:
    print('Warning: [{}] exists, skip it...'.format(output_sampled_revised_file))