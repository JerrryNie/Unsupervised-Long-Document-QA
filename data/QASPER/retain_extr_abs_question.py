"""Only retain extractive and abstractive questions for evaluation"""
import json
input_files = ['qasper-train-v0.3.json', 'qasper-dev-v0.3.json', 'qasper-test-v0.3.json']
output_files = ['qasper-train-v0.3-extr-abs.json',
                'qasper-dev-v0.3-extr-abs.json',
                'qasper-test-v0.3-extr-abs.json']

for input_file, output_file in zip(input_files, output_files):
    with open(input_file, 'r', encoding='utf-8') as f:
        qasper = json.load(f)
    for paper_id, paper in qasper.items():
        qas = []
        for qa in paper['qas']:
            answers = []
            for answer in qa['answers']:
                if answer['answer']['unanswerable'] is False and answer['answer']['yes_no'] is None:
                    answers.append(answer)
            qa['answers'] = answers
            if len(answers) > 0:
                qas.append(qa)
        paper['qas'] = qas
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qasper, f, indent=2, ensure_ascii=False)
