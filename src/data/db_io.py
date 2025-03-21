import lmdb
import pickle
import os
import numpy as np
from pathlib import Path
from glob import glob

from .pep_vocab import PepVocab

class DB_IO:
    def __init__(self, db_path, map_size=1024**2):
        self.db_path = db_path
        self.map_size = map_size
        self.env = None
        self.name2idx = {}
        
        self._init_db()
        
        self.vocab = PepVocab()
    
    def _init_db(self):
        self.env = lmdb.open(self.db_path, map_size=self.map_size)
    
    def _write_db(self, mgf_blocks: dict):
        with self.env.begin(write=True) as txn:
            for idx, (file_name, block) in enumerate(mgf_blocks.items()):
                self.name2idx[file_name] = idx
                block = self._filter_spec(block)
                txn.put(str(idx).encode(), pickle.dumps(block))
    
    def _read_db(self, file_name: str):
        idx = self.name2idx[file_name]
        with self.env.begin() as txn:
            block = txn.get(str(idx).encode())
            if block is not None:
                return pickle.loads(block)
            else:
                return None

    
    def _close_db(self):
        self.env.close()
    
    def _filter_spec(self, block: dict):
        """
        Filter out low intensity peaks and peaks outside the range of 200-2000 m/z.
        在这里处理数据，维持原始数据的格式，方便未来进一步处理

        Args:
            block (dict): MGF block.

        Returns:
            dict: Filtered MGF block.
        """
        block['spec'] = np.vstack((block['m/z'], block['intensity'])).T
        block['seq_idx'] = self.vocab[self.vocab.split_seq(block['seq'].upper())]
        return block
    
    def write_mgf_to_db(self, mgf_path):
        """
        Write mgf files to lmdb database.

        Args:
            mgf_path (str): Path to mgf files.
        """
        raw_mgf_blocks = self.read_mgf(mgf_path)
        self._write_db(raw_mgf_blocks)
    
    def read_mgf(self, mgf_path):   
        raw_mgf_blocks = {}
        for file in glob(os.path.join(mgf_path,'*.mgf')):
            with open(file) as f:
                for line in f:
                    if line.startswith('BEGIN IONS'):
                        product_ions_moverz = []
                        product_ions_intensity = []
                    elif line.startswith('TITLE'):
                        file_name = line.strip().split('=')[-1]
                    elif line.startswith('PEPMASS'):
                        pep_mass = float(line.strip().split('=')[-1])
                    elif line.startswith('CHARGE'):
                        charge = float(line.strip().split('=')[-1][0])
                    elif line.startswith('SEQ'):
                        seq = line.strip().split('=')[-1]
                        seq = seq.replace(' ', '').replace("L", "I")
                    elif line[0].isnumeric():
                        product_ion_moverz, product_ion_intensity = line.strip().split(' ')
                        product_ions_moverz.append(float(product_ion_moverz))
                        product_ions_intensity.append(int(float(product_ion_intensity)))
                    elif line.startswith('END IONS'):
                        raw_mgf_blocks[file_name] = {'m/z':np.array(product_ions_moverz),
                                                    'intensity':np.array(product_ions_intensity),
                                                    'pep_mass': np.array(pep_mass),
                                                    'charge': np.array(charge),
                                                     'seq': seq}
        return raw_mgf_blocks
    
    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = str(idx).encode()
            value = txn.get(key)
            if value is not None:
                return pickle.loads(value)
            else:
                return None
    
    def __len__(self):
        with self.env.begin(write=False) as txn:
            return txn.stat()['entries']
    
    