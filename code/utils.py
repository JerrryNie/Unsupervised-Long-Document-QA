import nltk
from tqdm import tqdm
import copy
import json
from contextlib import contextmanager
import tempfile
import tarfile
import logging
import shutil
import os
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.common.params import Params
from qasper_baselines import model
logger = logging.getLogger(__name__)
CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"


def get_weights_path(serialization_dir):
    weights_path = os.path.join(serialization_dir, _WEIGHTS_NAME)
    # Fallback for serialization directories.
    if not os.path.exists(weights_path):
        weights_path = os.path.join(serialization_dir, _DEFAULT_WEIGHTS)
    return weights_path


def extracted_archive(resolved_archive_file):
    tempdir = tempfile.mkdtemp()
    logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
    with tarfile.open(resolved_archive_file, "r:gz") as archive:
        archive.extractall(tempdir)
    return tempdir

def _load_model(config, weights_path, serialization_dir, cuda_device):
    return Model.load(
        config,
        weights_file=weights_path,
        serialization_dir=serialization_dir,
        cuda_device=cuda_device,
    )


def load_allennlp_ckpt(ckpt_file: str, cuda_device: int = -1):
    serialization_dir = extracted_archive(ckpt_file)
    weights_path = get_weights_path(serialization_dir)
    # Load config
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), "")
    print('serialization_dir: {}'.format(serialization_dir))
    # Instantiate model and dataset readers. Use a duplicate of the config, as it will get consumed.
    model = _load_model(config.duplicate(), weights_path, serialization_dir, cuda_device)
    model = model.transformer
    if serialization_dir is not None:
        print(f"removing temporary unarchived model dir at {serialization_dir}")
        shutil.rmtree(serialization_dir, ignore_errors=True)
    return model


def text_clean(text):
    if isinstance(text, str):
        cleaned_txt = text.replace('</s>', ' ').replace('<s>', ' ').replace("\n", " ").strip()
        cleaned_txt = ' '.join(cleaned_txt.split())
        cleaned_txt = ' '.join(nltk.word_tokenize(cleaned_txt))
        cleaned_txt = cleaned_txt.strip()
        idx = cleaned_txt.find(' \'')
        while idx != -1:
            cleaned_txt = cleaned_txt[:idx] + cleaned_txt[idx + 1:]
            idx = cleaned_txt.find(' \'')

        return cleaned_txt if cleaned_txt else '[ PAD ] .'
    elif isinstance(text, list):
        cleaned_txts = []
        for txt in text:
            if isinstance(txt, str):
                cleaned_txt = txt.replace('</s>', ' ').replace('<s>', ' ').replace("\n", " ").strip()
                cleaned_txt = ' '.join(cleaned_txt.split())
                cleaned_txt = ' '.join(nltk.word_tokenize(cleaned_txt))
                cleaned_txt = cleaned_txt.strip()
                idx = cleaned_txt.find(' \'')
                while idx != -1:
                    cleaned_txt = cleaned_txt[:idx] + cleaned_txt[idx + 1:]
                    idx = cleaned_txt.find(' \'')
                cleaned_txts.append(cleaned_txt if cleaned_txt else '[ PAD ] .')
            else:
                print('Warning: find nonstr, convert to empty str')
                cleaned_txts.append('[ PAD ] .')
        return cleaned_txts
    else:
        print('Warning: find nonstr, convert to empty str')
        return '[ PAD ] .'


def convert_qasper_context(input_file, stop_spans, existing_papers = None, start_idx = -1):
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    paper_id_to_paras = {}
    paper_ids = []
    debug_cnt = 0
    paper_start_idx = 0
    for paper_id, article in tqdm(input_data.items(), total=len(input_data), desc='read data...'):
        paper_start_idx += 1
        if start_idx > paper_start_idx - 1:
            continue
        if existing_papers is not None and paper_id in existing_papers:
            continue
        paper_ids.append(paper_id)
        sections = []
        for section_info in article['full_text']:
            section_name = text_clean(section_info['section_name']) if section_info['section_name'] is not None else '[ PAD ] .'
            if ':::' not in section_name:
                stop_spans.add(section_name)
            paras = []
            for p in text_clean(section_info['paragraphs']):
                paras.append(p)
            sections.append({
                'section_name': section_name,
                'paragraphs': paras
            })
        paper_id_to_paras[paper_id] = sections
        debug_cnt += 1
        if debug_cnt == 100:
            break
    return paper_id_to_paras, paper_ids
