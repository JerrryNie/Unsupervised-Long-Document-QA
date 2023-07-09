export CUDA_VISIBLE_DEVICES=$1

allennlp train training_config/led_base_smaller_context_plus_txt.jsonnet -s output_with_evidence_plus_txt --include-package qasper_baselines
