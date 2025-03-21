import torch
from typing import Optional, Tuple
from .db_io import DB_IO
import numpy as np
import spectrum_utils.spectrum as sus
from torch.utils.data import Dataset
from glob import glob

class MGFDataset(Dataset):
    '''
    不做mgf文件处理，在db_io初始化后 ,直接管理DB_IO对象
    '''
    def __init__(self, db_io: DB_IO):
        self.db_io = db_io                 # 不可序列化对象，dataloader不能多进程加载数据
        self.num_spectra = len(self.db_io) 
        
    def __len__(self):
        return self.num_spectra
    
    def __getitem__(self, idx):
                            
        mgf_block = self.db_io[idx]
        spectrum, precursor_mz, precursor_charge, peptide = mgf_block['spec'], mgf_block['pep_mass'], mgf_block['charge'], mgf_block['seq_idx']
        print(spectrum.shape)          
        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        precursor_mz = torch.tensor(precursor_mz, dtype=torch.float32)
        precursor_charge = torch.tensor(precursor_charge, dtype=torch.float32) 
       
        return spectrum, precursor_mz, precursor_charge, peptide
    
    