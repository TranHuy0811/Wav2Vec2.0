import torchaudio
from torch.utils.data import Dataset
import h5py


def Processing_Dataset(audio_path, transcript, cfg, processor):
    waveform, initial_sr = torchaudio.load(audio_path)
    if initial_sr != cfg['dataset_config']['sample_rate']:
        waveform = torchaudio.transforms.Resample(orig_freq=initial_sr, new_freq=cfg['dataset_config']['sample_rate'])(waveform)
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.squeeze()
    labels = processor(text=transcript.lower(), return_tensors="pt").input_ids.squeeze()

    return input_values, labels



class CustomDataset(Dataset):
    def __init__(self, txt_path, cfg, processor):
        self.cfg = cfg
        self.processor = processor
        with open(txt_path, "r", encoding="utf-8") as file:
            self.samples_path = file.readlines()
        self.length = len(self.samples_path)


    def __getitem__(self, idx):
        audio_path, transcript = self.samples_path[idx].strip().split('\t')
        input_values, labels = Processing_Dataset(audio_path, transcript, self.cfg, self.processor)
        return {"input_values": input_values, "label_ids": labels}
    

    def __len__(self):
        return self.length
        
        



# My custom dataset used for training (have been processed and stored in h5 file)
class Myh5Dataset(Dataset):
    def __init__(self, file_path_array, cfg):
        self.cfg = cfg
        length = 0
        file_array = []
        for i in file_path_array:
            tmp = h5py.File(i,'r')
            file_array.append(tmp)
            length += tmp['Input_Values'].shape[0]
            
        self.file_array = file_array
        self.length = length
        del file_array, length


    def Compute(self,idx):
        prev_len = 0
        iter = -1
        for i in self.file_array:
            iter += 1
            cur_len = prev_len + i['Input_Values'].shape[0]
            if idx < cur_len:
                return iter, idx - prev_len
            prev_len = cur_len
        return "err", "Error"
            

    def __getitem__(self, idx):    
        numfile,idx_in_file = self.Compute(idx)
        
        return {"input_values": self.file_array[numfile]['Input_Values'][idx_in_file],
                "label_ids": self.file_array[numfile]['Labels'][idx_in_file]}
    

    def __len__(self):
        return self.length