import torch 
from torch import nn
import pytorch_lightning as pl


"""
Given a model, a loss function and optimizer, implement the methods train_step, validation_step and test_step for them 
Args:
    model (nn.Module): an instanced pytorch model
    loss_config (dict: dict from mconfig
    ptim_config (dict): "" ""
"""    
class LitModelWrapper(pl.LightningModule): 
    def __init__(self, model, loss_config, optim_config):
        super().__init__()
        self.model = model
        self.loss_fn = eval(loss_config['loss_fn'])()
        self.optim_config = optim_config
        #self.optimizers = self.configure_optimizers() #do we need this?

    def forward(self, batch):
        X, label = batch
        logits = self.model(X)
        return logits

    def training_step(self, batch, batch_idx):
        X, label = batch
        batch_size = X.shape[0]
        logits = self.model(X)
        loss = self.loss_fn(logits, label.ravel())
        ## use self.log to log data to whatever type of logger you want( logger is handled by pl::Trainer)
        self.log('train_loss', loss, batch_size=batch_size)
        opt = self.optimizers()
        self.log('learning_rate', opt.optimizer.param_groups[0]['lr'], batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx): ## these arguments are required for Lighting, but you dont need to use them 
        ## dont need to worry about with torch.no_grad() and model.eval(), Lighting handles it for you 
        X, label = batch
        batch_size = X.shape[0]
        logits = self.model(X)
        loss = self.loss_fn(logits, label.ravel())
        self.log(f"validation_loss", loss, sync_dist = True, batch_size=batch_size) # sync_dists = True makes sure metric is averaged across multiple gpus; if set false, only gives bck data from 0th process 
        
    def test_step(self, batch, batch_idx, dataset_idx):## same as validation_step function
        X, label = batch
        batch_size = X.shape[0]
        logits = self.model(X)
        loss = self.loss_fn(logits, label)
        self.log(f"test_loss", loss, sync_dist = True, batch_size=batch_size)

    def configure_optimizers(self):
        optim_fn = eval(self.optim_config['optim_fn'])
        optimizer = optim_fn(self.parameters(), **self.optim_config['optim_kwargs'])
        ### You can also use a learning rate scheduler here, but Ive commented it out for simplicity
        # sched_fn = eval(self.optim_config["scheduler"]) 
        # scheduler = sched_fn(optimizer,  **self.optim_config['scheduler_kwargs'])
        # scheduler_config = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        #     "name":"learning_rate"
        # }
        optimizer_dict = {"optimizer" : optimizer#, 
            #"lr_scheduler" : scheduler_config
         }
        return optimizer   # it was originally optimizer_dict
