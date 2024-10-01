import argparse
from util import load_config
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")



def Inference(args, cfg):
    model = Wav2Vec2ForCTC.from_pretrained(args.model_pretrained).to(device)
    model.eval()
    result = []

    with torch.no_grad():
        for file_path in args.input_files:
            waveform, initial_sr = torchaudio.load(file_path)
            if initial_sr != cfg['dataset_config']['sample_rate']:
                waveform = torchaudio.transforms.Resample(orig_freq=initial_sr, new_freq=cfg['dataset_config']['sample_rate'])(waveform)
                input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
                logits = model(input_values).logits
                pred_id = torch.argmax(logits, axis=-1)
                pred_transcript = processor.batch_decode(pred_id)[0]

                result.append(pred_transcript)
        
    return result    




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pretrained")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input_files", required=True,  nargs='+', help="List of input audio file paths")
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    return Inference(args, cfg)




if __name__ == "__main__":
    main()