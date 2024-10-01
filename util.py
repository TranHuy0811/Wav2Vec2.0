import yaml
import torch
import shutil
import numpy as np
import evaluate

wer_metric = evaluate.load("wer",trust_remote_code=True)



def load_config(cfg_path):
    with open(cfg_path) as file:
        return yaml.safe_load(file)




def initialize_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def remove_checkpoint(directory):
    shutil.rmtree(directory)



def compute_metric_wer(logits, labels, processor):
    pred_ids = np.argmax(logits, axis=-1)
    labels[labels == -100] = processor.tokenizer.pad_token_id

    pred_transcripts = processor.batch_decode(pred_ids)
    label_transcripts = processor.batch_decode(labels, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_transcripts, references=label_transcripts)
    return wer