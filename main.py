from autotransformers import DatasetConfig, ModelConfig, AutoTrainer, ResultsPlotter
from transformers import EarlyStoppingCallback
from datasets import load_dataset

import pandas as pd
import numpy as np

dataset = load_dataset("rotten_tomatoes")

hp_space_results = {
    "learning_rate": 1.132933165272574e-05,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "warmup_ratio": 0.04751388086243615,
    "weight_decay": 3.47814371740729e-08,
    "adam_epsilon": 1.0084904907887147e-08
}

fixed_train_args = {
    "do_train": True,
    "logging_strategy": "epoch",
    "evaluation_strategy" : "epoch",
    "eval_steps": 1,
    "save_steps": 1,
    "logging_steps": 1,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "seed": 42,
    "fp16": True,
    "load_best_model_at_end": True,
    'optim' : 'adafactor'
}

fixed_train_args.update(hp_space_results)

default_args_dataset = {
    "seed": 44,
    "direction_optimize": "maximize",
    "metric_optimize": "eval_f1-score",
    "retrain_at_end": False,
    "fixed_training_args": fixed_train_args
}

rotten_tomatoes_config = default_args_dataset.copy()
rotten_tomatoes_config.update(
    {
        "dataset_name": "rotten_tomatoes",
        "alias": "rotten_tomatoes",
        "task": "classification",
        "label_col": "label",
        "text_field": "text",
        "hf_load_kwargs": {"path": "rotten_tomatoes"}
    }
)

rotten_tomatoes_config = DatasetConfig(**rotten_tomatoes_config)

debertav3_config = ModelConfig(
    name="microsoft/deberta-v3-large",
    save_name="debertabase",
    n_trials=1,
    additional_params_tokenizer={"model_max_length": 512}
)

autotrainer = AutoTrainer(
    model_configs=[debertav3_config],
    dataset_configs=[rotten_tomatoes_config],
    metrics_dir="rottentomatoes_metrics",
    hp_search_mode="fixed"
)

def tokenize_function(examples):
    """
    Tokenizes the text examples using the tokenizer from the AutoTrainer instance.

    This function is intended to be used with the `map` method of a Hugging Face dataset. 
    It tokenizes the text, adds padding to the maximum length specified, and ensures that 
    the tokenization is truncated to the max length if it exceeds it.

    Parameters:
    - examples (dict): A dictionary containing the texts to be tokenized, with the key "text"
      pointing to a list of string texts.

    Returns:
    - dict: A dictionary with the tokenized inputs suitable for input to a model. The keys of the
      dictionary correspond to the model's expected input names.
    """
    return autotrainer.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def main():
  """
  Main function to execute the training and prediction process using the AutoTrainer.

  This function runs the AutoTrainer with the configuration set for the Deberta-v3 model
  and the Rotten Tomatoes dataset. After training, it performs prediction on the test set,
  calculates the f1-score, and saves the prediction results to a CSV file.

  The predictions are saved in a file named 'results.csv', with two columns: 'index' for the
  example index in the test set, and 'pred' for the predicted label (1s and 0s).
    
  Outputs are printed to the console, including the f1-score achieved on the test set and
  the path to the saved CSV file with predictions.
  """
  # Train the model.
  _ = autotrainer()

  # Load test set and evaluate results.
  test = load_dataset('rotten_tomatoes', split='test')
  test = test.map(tokenize_function, batched=True)
  results = autotrainer.trainer.predict(test)

  print(f"The resulting f1-score is {round(results.metrics['test_f1-score'], 3)}")

  def compute_metrics(pred):
    logits = pred
    return np.argmax(logits, axis=-1)

  preds = compute_metrics(results.predictions)

  df = pd.DataFrame({
      'index': np.arange(len(preds)),
      'pred': preds
  })

  csv_path = 'results.csv'

  df.to_csv(csv_path, index=False)

  print(f"Predictions saved to {csv_path}")

if __name__ == '__main__':
  main()