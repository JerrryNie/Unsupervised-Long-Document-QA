export CUDA_VISIBLE_DEVICES=$1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
allennlp train training_config/led_base_with_evidence_scaffold_ulqa_v3_nar.jsonnet -s output_full_ulqa_v3_nar --include-package qasper_baselines
