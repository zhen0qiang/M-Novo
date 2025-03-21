import torch
from torch.utils.data import DataLoader

from .mdataset import MGFDataset
from .db_io import DB_IO


class DataManage:
    def __init__(self, train_db_path=None,
                 test_db_path=None,
                 valid_db_path=None):
        
        self.train_db_path = train_db_path
        self.test_db_path = test_db_path
        self.valid_db_path = valid_db_path
        
        self.train_dataloader = None
        self.test_dataloader = None
        self.valid_dataloader = None
        
    def setup_db(self, db_path, map_size=1024**2):
        db_io = DB_IO(db_path, map_size)
        dataset = MGFDataset(db_io)
        return dataset
        
    def train_loader(self, batch_size=1, num_workers=0):
        train_dataset = self.setup_db(self.train_db_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, 
                                      shuffle=True, collate_fn=self.collate_fn)
        
        return train_dataloader
        
    def collate_fn(self, batch):
        spectrum, precursor_mz, precursor_charge, peptide = list(zip(*batch))
                
        spectrum = torch.nn.utils.rnn.pad_sequence(spectrum, batch_first=True)
        precursor_mz = torch.tensor(precursor_mz)
        precursor_charge = torch.tensor(precursor_charge)
        
        batch_size, seq_len, feature_size = spectrum.shape
        peptide = self.truncate_pad(peptide, seq_len)
        peptide = torch.tensor(peptide)
        
        
        # input_data = torch.vstack([spectrum, precursor_mz, precursor_charge])
        
        return spectrum, precursor_mz, precursor_charge, peptide
    
    def truncate_pad(self, line, num_steps, padding_token=0) -> list:

        if not isinstance(line[0], list):
            if len(line) > num_steps:
                return line[:num_steps]
            return line + [padding_token] * (num_steps - len(line))
        else:
            return [self.truncate_pad(l, num_steps, padding_token) for l in line]   # a list of list