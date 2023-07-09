from datasets import load_dataset
import hashlib
import json
from tqdm import tqdm
dataset = load_dataset("narrativeqa")
output_files = {'validation': 'narrativeqa-dev.json',
                'train': 'narrativeqa-train.json',
                'test': 'narrativeqa-test.json'}

for split, output_file in output_files.items():
    input_data = dataset[split]
    output_data = {}
    if split == 'train':
        qa_ids = set()
    else:
        qa_ids = None
    for item in tqdm(input_data, desc=split):
        paper_id = hashlib.md5(item['document']['url'].encode()).hexdigest()
        if paper_id not in output_data:
            output_data[paper_id] = {}
            output_data[paper_id]['full_text'] = [
            ]
            cnt_char = 0
            sec_pointer = 0
            cur_para = ''
            for p in item['document']['text'].split('\n\n'):
                p = p.split()
                if len(p) > 0 and cnt_char >= 5000:
                    output_data[paper_id]['full_text'].append({
                        'section_name': None,
                        'paragraphs': [cur_para]
                    })
                    # [0]['paragraphs'].append(cur_para)
                    cnt_char = 0
                    sec_pointer += 1
                    cur_para = ''
                if len(p) > 0 and cnt_char < 5000:
                    cur_para += ' ' + ' '.join(p[:5000])
                    cnt_char += len(' '.join(p[:5000]))
            output_data[paper_id]['qas'] = []
        question_txt = item['question']['text'].lower()
        answer_txts = [answer['text'].lower() for answer in item['answers']]
        qid = hashlib.md5((paper_id + question_txt).encode()).hexdigest()
        qa = {
            'question': question_txt,
            'question_id': qid,
            'answers': [{'answer': {'free_form_answer': answer_txt, 'evidence': []}} for answer_txt in answer_txts]
        }
        if split == 'train':
            answers = []
            for answer in qa['answers']:
                answer_txt = answer['answer']['free_form_answer']
                if (paper_id, question_txt, answer_txt) not in qa_ids:
                    qa_ids.add((paper_id, question_txt, answer_txt))
                    answers.append(answer)
            if len(answers) == 0:
                continue
            else:
                qa['answers'] = answers
        output_data[paper_id]['qas'].append(qa)
    print(len(output_data))
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
