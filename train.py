import argparse
import json
import os
from collections import deque
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime
import wandb
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config

from dataset import CustomDataset, Myh5Dataset
from util import (
    load_config, initialize_seed, compute_metric_wer, remove_checkpoint
)
from setup import (
    setup_optimizer, setup_scheduler, setup_dataloader
)

torch.backends.cudnn.benchmark=True

current_time = datetime.now()
formatted_time = current_time.strftime("%Y_%m_%d-%H_%M_%S")
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")




def Train(rank, cfg, args):
    num_gpus = cfg['env_config']['num_gpus']
    train_batch_size = cfg['training_config']['train_batch_size'] // num_gpus   # Batch Size Per Device
    eval_batch_size =  cfg['training_config']['eval_batch_size'] // num_gpus
    
    if num_gpus > 1 : 
        init_process_group(
            backend=cfg['env_config']['dist_config']['dist_backend'],
            init_method=cfg['env_config']['dist_config']['dist_url'],
            world_size=cfg['env_config']['dist_config']['world_size'] * cfg['env_config']['num_gpus'],
            rank=rank
        )
    device = torch.device('cuda:{:d}'.format(rank))



    '''  Define and load model, optimizer, scheduler '''

    if args.model_pretrained: 
        model = Wav2Vec2ForCTC.from_pretrained(args.model_pretrained)
    else: 
        with open('model_architecture.json', 'r') as json_file:
            loaded_config = json.load(json_file)
        loaded_config = Wav2Vec2Config(**loaded_config)
        model = Wav2Vec2ForCTC(config = loaded_config)

    model.freeze_feature_encoder()
    model.to(device)

    if num_gpus > 1 and torch.cuda.is_available():
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)

    

    optimizer = setup_optimizer(model, cfg)
    if args.optimizer_pretrained:
        optimizer.load_state_dict(torch.load(args.optimizer_pretrained, map_location=device))



    step = 0
    last_epoch = -1
    if args.scheduler_pretrained:
        scheduler_state_dict = torch.load(args.scheduler_pretrained, map_location=device)
        step = scheduler_state_dict['last_step']
        last_epoch = scheduler_state_dict['last_epoch']
    scheduler = setup_scheduler(optimizer, cfg, last_epoch)




    '''   Load Dataset   '''
    if cfg['dataset_config']['use_h5_file'] == False:
        train_dataset = CustomDataset(cfg['dataset_config']['train_file_path'], cfg, processor)
    else:
        train_dataset = Myh5Dataset(cfg['dataset_config']['train_file_path'], cfg)

    train_dataloader = setup_dataloader(train_dataset, cfg, train_batch_size, train=True)


    if cfg['dataset_config']['use_h5_file'] == False:
        eval_dataset = CustomDataset(cfg['dataset_config']['eval_file_path'], cfg, processor)
    else:
        eval_dataset = Myh5Dataset(cfg['dataset_config']['eval_file_path'], cfg)

    eval_dataloader = setup_dataloader(eval_dataset, cfg, eval_batch_size, train=False)


    ''' Initialize Wandb for Logging ''' 
    if rank == 0: 
        wandb.init(project="Wave2Vec2.0")



    '''    Training    '''
    model.train()
    scaler = GradScaler(enabled=cfg['training_config']['mixed_precision'])
    optimizer.zero_grad(set_to_none=True)
    cur_grad_accu_step = 0
    accu_loss = 0
    checkpoint_queue = deque()

    for epoch in range(cfg['training_config']['train_epochs']):
        for batch in train_dataloader:
            cur_grad_accu_step += 1

            input_values = batch['input_values'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg['training_config']['mixed_precision']):
                loss = model(input_values=input_values, labels=labels).loss
            scaler.scale(loss).backward()
            
            accu_loss += loss.item()

            if torch.isnan(loss).any():
                raise ValueError("NaN values found in loss")


            if cur_grad_accu_step % cfg['training_config']['train_gradient_accumulation'] == 0: # Done 1 step of gradient accumulation
                accu_loss = accu_loss / cfg['training_config']['train_gradient_accumulation']
                cur_grad_accu_step = 0
                step += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training_config']['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
                if rank == 0:
                    
                    # Stdout
                    if step % cfg['env_config']['stdout_interval'] == 0:
                        print('Step: {:d}, Loss {:4.6f}'.format(step, accu_loss))


                    # Saving Checkpoint
                    if step % cfg['env_config']['checkpoint_interval'] == 0:

                        ckp_path = os.path.join(args.exported_path, f"{step:08d}")
                        os.makedirs(ckp_path, exist_ok=True)
                        

                        # Remove Checkpoint
                        if len(checkpoint_queue) == cfg['env_config']['checkpoint_saving_limit']:
                            remove_checkpoint(checkpoint_queue[0])
                            checkpoint_queue.popleft()
                        checkpoint_queue.append(ckp_path)


                        # Save
                        cur_exp_path = os.path.join(ckp_path, "ckp_model")
                        model.save_pretrained(cur_exp_path)

                        cur_exp_name = os.path.join(ckp_path, "optimizer.pth")
                        torch.save(optimizer.state_dict(), cur_exp_name)

                        cur_exp_name = os.path.join(ckp_path, "scheduler.pth")
                        torch.save({
                            "last_step": step,
                            "last_epoch": epoch 
                        },cur_exp_name)


                    # Logging
                    if args.wandb_key and step % cfg['env_config']['log_interval'] == 0:
                        wandb.log({"Training/Loss": accu_loss}, step=step)
                        wandb.log({"Training/Learning_Rate": scheduler.get_last_lr()[0]}, step=step)

                    
                    # Evaluation
                    if args.wandb_key and step % cfg['env_config']['eval_interval'] == 0:
                        model.eval()
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            eval_mean_loss = 0
                            eval_mean_wer = 0
                            for cnt,eval_batch in enumerate(eval_dataloader):
                                input_values = eval_batch['input_values'].to(device, non_blocking=True)
                                labels = eval_batch['labels'].to(device, non_blocking=True)

                                out = model(input_values=input_values, labels=labels)
                                eval_loss = out.loss
                                eval_logits = out.logits
                                
                                eval_mean_loss += eval_loss.item()
                                eval_mean_wer += compute_metric_wer(eval_logits.detach().cpu().numpy(), labels.cpu().numpy(), processor)
                            
                            eval_mean_loss = eval_mean_loss / (cnt+1)
                            eval_mean_wer = eval_mean_wer / (cnt+1)

                            print('Steps : {:d}, Eval Loss: {:4.3f}, WER:'.format(step, eval_mean_loss, eval_mean_wer))
                            wandb.log({"Eval/Loss": eval_mean_loss}, step=step)
                            wandb.log({"Eval/WER": eval_mean_wer}, step=step)
                        
                        model.train()

                if step % cfg['training_config']['lr_decay_interval'] == 0:
                    scheduler.step()

                accu_loss = 0


        if rank == 0:
            print(f"Finished epoch {epoch}.")


    # Save model when finished training
    if rank == 0:
        print("Training Finished!") 

        res_path = os.path.join(args.exported_path, "result")
        os.makedirs(res_path, exist_ok=True)

        cur_exp_path = os.path.join(res_path, "model")
        model.save_pretrained(cur_exp_path)

        cur_exp_name = os.path.join(res_path, "optimizer.pth")
        torch.save(optimizer.state_dict(), cur_exp_name)

        cur_exp_name = os.path.join(res_path, "scheduler.pth")
        torch.save({
            "last_step": step,
            "last_epoch": epoch 
        },cur_exp_name)       

            



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "config.yaml")
    parser.add_argument("--exported_path", default = f"export/{formatted_time}")
    parser.add_argument("--model_pretrained", default = None)
    parser.add_argument("--optimizer_pretrained", default = None)
    parser.add_argument("--scheduler_pretrained", default = None)
    parser.add_argument("--wandb_key", default = None)
    args = parser.parse_args()


    cfg = load_config(args.config)
    initialize_seed(cfg['env_config']['seed'])
    os.makedirs(args.exported_path, exist_ok=True)



    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus < cfg['env_config']['num_gpus']:
            warnings.warn(
                f"Warning: The actual number of GPUs ({num_available_gpus}) is less than the one you set on config ({cfg['env_config']['num_gpus']}). We're going to change it to {num_available_gpus}", UserWarning
            )
            cfg['env_config']['num_gpus'] = num_available_gpus
    else:
        raise RuntimeError("Need GPU for training!") 
    

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    
    if cfg['env_config']['num_gpus'] > 1:
        mp.spawn(Train, nprocs=cfg['env_config']['num_gpus'], args=(cfg, args))
    else :
        Train(0, cfg, args)




if __name__ == "__main__":
    main()