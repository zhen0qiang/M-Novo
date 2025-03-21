from torch.optim import Adam

from src.data.DataManage import DataManage
from src.models.model import make_model
from src.models.loss import Loss

class Task:
    def __init__(self, cfg):
        self.cfg = cfg
        self.datamanage =DataManage(self.cfg.db.train_db_path, self.cfg.db.test_db_path, self.cfg.db.valid_db_path)
        self.initialize()
    
    def initialize(self):
        '''导入模型和数据'''
        self.model = make_model(2, 25, 2)
        
        self.loss_fn = Loss('CrossEntropyLoss')
        self.opt = Adam(self.model.parameters(), lr=self.cfg.train.lr)
        
        self.train_data_loader = self.datamanage.train_loader(batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers)
    
    def train_loader(self):
        '''准备所有数据'''
        train_loader = self.datamanage.train_loader(batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
        return train_loader
    
    def train(self):
        print('开始训练')
        for epoch in range(self.cfg.train.num_epochs):
            for i, (spectrum, precursor_mz, precursor_charge, peptide) in enumerate(self.train_data_loader):
                print(i, spectrum.shape, precursor_mz.shape, precursor_charge.shape, peptide.shape)
                output = self.model(spectrum)
                print(output.shape)
                loss = self.loss_fn(output, peptide)
                print('[Epoch %d] [Batch %d] [Loss %.5f]' % (epoch, i, loss.item()))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
    
    def inference(self):
        pass
    