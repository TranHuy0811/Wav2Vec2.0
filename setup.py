import torch.optim as optim
from torch.utils.data import DistributedSampler, DataLoader
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

def setup_optimizer(model, cfg):
    learning_rate = cfg['training_config']['learning_rate']
    betas = (cfg['training_config']['adam_b1'],cfg['training_config']['adam_b2'])
    weight_decay = cfg['training_config']['weight_decay']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer




def setup_scheduler(optimizer, cfg, last_epoch=-1):
    lr_decay = cfg['training_config']['lr_decay']
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay, last_epoch=last_epoch)
    return scheduler




def data_collator(features):
    input_features = [{"input_values": feature["input_values"]} for feature in features]
    label_features = [{"input_ids": feature["label_ids"]} for feature in features]

    batch = processor.pad(
        input_features,
        padding = True,
        return_tensors = "pt"
    )
    with processor.as_target_processor():
        batch_labels = processor.pad(
            label_features,
            padding = True,
            return_tensors = "pt"
        )
    labels = batch_labels["input_ids"].masked_fill(batch_labels.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch



def setup_dataloader(dataset, cfg, batch_size, train=False):
    if cfg['env_config']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_config']['train_epochs']) 
    else:
        sampler = None
    num_workers = cfg['env_config']['num_workers']

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=(sampler is None) and train,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collator
    )
    return dataloader