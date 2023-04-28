# for command line args and json config file parsing
import sys
import json
# for linking .py files
from model import *
from lightning_modules import *
from data import train_dataset, val_dataset, test_dataset
# for logging results
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


args = sys.argv[1:]
if len(args) == 1:
    ## using a run 
    configfile =  f"{sys.argv[1]}"
    with open(configfile) as m_stream:
        mconfig = json.load(m_stream)
else:
    configfile =  f"/Users/emmachen/Documents/aqlab/uniprot-nn/models/configs/base_model.json"   # not sure if this works
    with open(configfile) as m_stream:
        mconfig = json.load(m_stream)



## Load data 
data_conf=mconfig['data']
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=data_conf['batch_size'], shuffle=True)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=data_conf['batch_size'], shuffle=True) 
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=data_conf['batch_size'], shuffle=True) 

## instance model 
_model = eval(mconfig['model_fn'])
litmodel = LitModelWrapper(model = _model(mconfig), loss_config= mconfig['loss'], optim_config = mconfig['optim'])



## instance wandb logger object
plg= WandbLogger(project = mconfig['wandb_project'],
                 entity = 'emmazchen', 
                 config=mconfig) ## include your run config so that it gets logged to wandb 
plg.watch(litmodel) ## this logs the gradients for your model 

## add the logger object to the training config portion of the run config 
trainer_conf = mconfig['training']
trainer_conf['logger'] = plg

## pytorch lightning saves the best checkpoint of your model by default,
## but I like to save every checkpoint, which this lets you do 
checkpoint_cb = ModelCheckpoint(save_top_k=-1, every_n_epochs = None,every_n_train_steps = None, train_time_interval = None)
trainer_conf['callbacks'] = [checkpoint_cb]
trainer_conf['devices'] = 1
trainer = pl.Trainer(**trainer_conf)

if mconfig['dryrun']: ## the dry run parameter lets you check if everythign can be loaded properly 
    print("Successfully loaded everything. Quitting")
    sys.exit()

trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders=val_dl) ## this starts training 

out = trainer.predict(litmodel, dataloaders = test_dl)



