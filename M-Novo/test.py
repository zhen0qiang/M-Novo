from src.data.mdataset import MGFDataset
from src.data.db_io import DB_IO
from src.data.DataManage import DataManage
from src.data.pep_vocab import PepVocab

from src.models.model import make_model
from src.tasks.task import Task

def test_for_db_io():
    import pickle
    db_path = './src/dataset/test.lmdb'
    mgf_path = './src/dataset/'
    db_io = DB_IO(db_path)
    db_io.write_mgf_to_db(mgf_path)
    with db_io.env.begin(write=False) as txn:
        for i, (key, value) in enumerate(txn.cursor()):
            print(key)
            v = pickle.loads(value)
            print(v['seq'])
            if i == 5:
                break
    print(db_io[3])
    print(db_io.env.stat())
    db_io._close_db()

def test_for_mdataset():
    
    db_path = './src/dataset/dataset_test.lmdb'
    mgf_path = './src/dataset/'
    db_io = DB_IO(db_path)
    db_io.write_mgf_to_db(mgf_path)
    
    dataset = MGFDataset(db_io)
    
    for i in range(len(dataset)):
        print(dataset[i])

def test_for_datamanage():
    
    data_manage = DataManage(train_db_path='./src/dataset/test.lmdb')
    train_loader = data_manage.train_loader(batch_size=2, num_workers=0)
    for i, (spectrum, precursor_mz, precursor_charge, peptide) in enumerate(train_loader):
        print(i, spectrum.shape, precursor_mz.shape, precursor_charge.shape, peptide)

def test_for_model():
    import torch
    import torch.nn as nn
    import numpy as np
    V = 10
    x = torch.randint(0, 10, (10, 20, 10), dtype=torch.float32)
    # x = torch.LongTensor(x)
    
    model = make_model(V, V, 2)
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    cri = nn.CrossEntropyLoss()
    
    for i in range(10):
        opt.zero_grad()
        out = model(x)
        loss = cri(out.view(-1, 10), torch.argmax(x, -1).view(-1))
        loss.backward()
        opt.step()
        
    
    print(out.shape)

def test_for_task():
    
    cfg_path = './src/config/config.yaml'
    from omegaconf import DictConfig, OmegaConf
    
    cfg = OmegaConf.load(cfg_path)
    print(OmegaConf.to_yaml(cfg))
    
    task = Task(cfg)
    task.train()

def test_for_pep_vocab():
    test_data = 'PEPTIDE'
    test_data_batch = ['PEPTIDE', 'PEPTIDE', 'PEPTIDE', 'PEPTIDE', 'PEPTIDE']
    
    vocab = PepVocab()
    
    test_data = vocab.split_seq(test_data)
    test_data_batch = vocab.split_seq(test_data_batch)
    
    vocab.add_special_token(['<SOS>', '<EOS>'])
    test_data_idx = vocab[test_data]
    test_data_batch_idx = vocab[test_data_batch]
    print(test_data_idx)
    print(test_data_batch_idx)
    
    print(vocab.to_tokens(test_data_idx))
    print(vocab.to_tokens(test_data_batch_idx))
    

if __name__ == '__main__':
    # test_for_db_io()
    # test_for_mdataset()
    # test_for_datamanage()    
    # test_for_model()
    # test_for_pep_vocab()
    test_for_task()