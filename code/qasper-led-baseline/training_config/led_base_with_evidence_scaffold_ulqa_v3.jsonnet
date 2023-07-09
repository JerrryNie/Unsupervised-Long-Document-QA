local transformer_model = "allenai/led-base-16384";
local epochs = 5;
local batch_size = 1;
local num_gradient_accumulation_steps = 4;

local train_data_path = "../ulqa_data_v3/ulqa_qasper_train_10000.json";
local dev_data_path = "../../data/QASPER/qasper-dev-v0.3-extr-abs.json";

// local training_data_size = 15435;
// local num_gpus = 1;


{
    "dataset_reader": {
        // "type": "sharded",
        // "base_reader": {
          "type": "qasper",
          "transformer_model_name": transformer_model,
          "max_document_length": 13000,
          "for_training": true,
        // }
    },
    "validation_dataset_reader": {
        "type": "qasper",
        "transformer_model_name": transformer_model,
	"max_document_length": 13000,
	"for_training": false,
    },
    "train_data_path": train_data_path,
    "validation_data_path": dev_data_path,
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "qasper_baseline",
        "transformer_model_name": transformer_model,
	// "attention_window_size": 1536,
  "attention_window_size": 640,
	"gradient_checkpointing": true,
	"use_evidence_scaffold": true,
	"attention_dropout": 0.1,
    },
    "data_loader": {
        "batch_size": batch_size,
    },
    "trainer": {
      "optimizer": {
        "type": "adam",
        "lr": 3e-5,
      },
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": epochs,
        "cut_frac": 0.1,
        // "num_steps_per_epoch": std.ceil(training_data_size / (batch_size * num_gradient_accumulation_steps * num_gpus)),
      },
      "callbacks": [
	{"type": "tensorboard"},
      ],
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "validation_metric": "+answer_f1",
      "enable_default_callbacks": false,
      "use_amp": true,
      "cuda_device": 0,
      "run_confidence_checks": false,
    },
    "pytorch_seed": 15371,
    "distributed": {
      "cuda_devices": [0, 1, 2, 3],
    },
}
