export CUDA_VISIBLE_DEVICES=$1
RETRIEVED_CONTEXT=../../data/NarrativeQA/narrativeqa-test.json
OUTPUT_FILE=ulqa_test_NarrativeQA.txt

allennlp evaluate output_full_ulqa_v3_nar/model.tar.gz \
    ${RETRIEVED_CONTEXT} --output-file ${OUTPUT_FILE} --cuda-device 0 --include-package qasper_baselines
