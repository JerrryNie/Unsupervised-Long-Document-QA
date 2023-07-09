export CUDA_VISIBLE_DEVICES=$1
RETRIEVED_CONTEXT=../../data/QASPER/qasper-test-v0.3-extr-abs.json
OUTPUT_FILE=ulqa_test_QASPER.txt

allennlp evaluate ./output_full_ulqa_v3/model.tar.gz \
    ${RETRIEVED_CONTEXT} --output-file ${OUTPUT_FILE} --cuda-device 0 --include-package qasper_baselines
