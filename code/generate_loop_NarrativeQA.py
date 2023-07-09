import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
os.environ['DATA_RANGE'] = sys.argv[2]
os.environ['AGGREGATOR'] = sys.argv[3]
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

while True:
    cmd = 'python span_collector_NarrativeQA.py {} {} {}'.format(
        sys.argv[1], sys.argv[2], sys.argv[3]
    )
    os.system(cmd)
