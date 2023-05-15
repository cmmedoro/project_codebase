
import torch
import numpy as np
import torchvision.models
import pytorch_lightning as pl
from torchvision import transforms as tfm
from pytorch_metric_learning import losses
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import parser1
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
#other libraries to import
import torch.nn as nn
import torch.nn.functional as F
import loss_miner as lm
import aggregators as ag
import self_modules as sm
#libraries for Proxy implementation
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler
import faiss
import random

class ProxySamplerVersione2(Sampler):

    first_epoch=0

    def __init__(self, dataset, batch_size, generator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = generator
        #Take the bank you have defined at the end of the previous epoch(in inference epoch end)
        #compute the final averages and instantiate the index
        global bank
        bank.computeavg()
        self.proxies= Proxies(bank)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
    def __iter__(self):
        if first_epoch==0:
            first_epoch=1
            for _ in range( len(self.dataset)// self.batch_size):
                yield from torch.randperm(self.batch_size, generator=self.generator).tolist()
            yield from torch.randperm(self.batch_size, generator=self.generator).tolist()[:len(self.dataset) % self.batch_size]
        else:
            while bank.__len__()>self.batch_size:
                randint = random.choice(bank.getkeys)
                #take neareast neighbors of the random place as selected places for the new batch
                #then remove selected places both from bank and from index
                indexes= self.proxies.getproxies(rand_index=randint, batch_size=self.batch_size)
                bank.remove_places(indexes)
                self.proxies.remove_places(indexes)
                yield indexes
            yield np.array(bank.getkeys)
            
    def __len__(self):
        return self._len
    

class ProxySampler(Sampler):

    first_epoch=0
    def __init__(self, dataset, batch_size, generator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = generator
        #Take the bank you have defined at the end of the previous epoch(in inference epoch end)
        #compute the final averages and instantiate the index
        global bank
        #bank.computeavg()
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
    def __iter__(self):
        if first_epoch==0:
            first_epoch=1
            for _ in range( len(self.dataset)// self.batch_size):
                yield from torch.randperm(self.batch_size, generator=self.generator).tolist()
            yield from torch.randperm(self.batch_size, generator=self.generator).tolist()[:len(self.dataset) % self.batch_size]
        else:
            bank.computeavg()
            self.proxies= Proxies(bank)
            batches=[]
            while bank.__len__()>self.batch_size:
                randint = random.choice(bank.getkeys())
                #take neareast neighbors of the random place as selected places for the new batch
                #then remove selected places both from bank and from index
                indexes= self.proxies.getproxies(rand_index=randint, batch_size=self.batch_size)
                bank.remove_places(indexes)
                self.proxies.remove_places(indexes)
                batches.append(indexes.tolist())
            batches.append(bank.getkeys())    
            return iter(batches)
        """Sampler usedas model:
        combined = list(first_half_batches + second_half_batches)
        combined = [batch.tolist() for batch in combined]
        random.shuffle(combined)
        return iter(combined)"""
            
    def __len__(self):
        return self._len

class ProxyHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=256,):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimred= nn.Linear(in_channels, out_channels)  #(512, 256)
        self.norm=ag.L2Norm()#Ragionare bene su quale dimensione devo andare ad agire
        #di default la dimensione è la 1
        #dovrei ricevere in input qualcosa che ha come dimensione[0] 256 (le immagini di un batch)
        #e come dimensione[1] 512, ovvero descriptors_dim

    def forward(self, x):
        x = self.dimred(x)#dimensionality reduction
        x = self.norm(x)
        return x
    

class ProxyBank():
    #E' un dizionario in cui ad ogni chiave, indice di un luogo, viene associato il tensore che sarà il rappresentante compatto di quel luogo,
    # ottenuto come media delle feature map ottenute da immagini di quel luogo, opportunamente ridimensionate
    def __init__(self):
        self.proxybank= {}

    def adddata(self, compact_descriptors, labels):
        #ad ogni batch della rete neurale dobbiamo aggiungere i nuovi descrittori
        for compact_descriptor, label in zip(compact_descriptors, labels):
            label=int(label)
            if label in self.proxybank.keys():
                self.proxybank[label][0]+=compact_descriptor
                self.proxybank[label][1]+=1
            else: 
                self.proxybank[label]=[compact_descriptor,1]
    
    def computeavg(self):
        #finita una epoch calcoliamo i rappresentanti compatti di ogni luogo
        for el in self.proxybank.values():
            el[0]=el[0]/el[1]
    
    def getdict(self):
        return self.proxybank
        
    def getkeys(self):
        return list(self.proxybank.keys())
    
    def remove_places(self, list_index):
      for el in list_index:
        self.proxybank.pop(el)

    #Da qui in giù le cose non ervono più ma le ho fatte e le lascio
    def __getitem__(self, key):
        return self.proxybank[key][0]
    def __len__(self):
        return len(self.proxybank)

class Proxies():
    #indicizziamo i rappresentanto compatti ottenut in proxyhead ai fini di poterli più facilmente confrontare tramite knn
    def __init__(self, proxybank):
        self.proxybank=proxybank.__getdict__()#dopo inizializzazione non viene più modificato
        self.places=proxybank.__getkeys__()#dopo inizializzazione non viene più modificato
        self.proxies=np.array([self.proxybank[key][0].numpy().astype(np.float32) for key in self.places])#dopo inizializzazione non viene più modificato
        support_index = faiss.IndexFlatL2(self.proxies.shape[1])
        self.proxy_faiss_index = faiss.IndexIDMap(support_index)
        self.proxy_faiss_index.add_with_ids(self.proxies, self.places)
        
    def getproxies(self, rand_index, batch_size):
        _,indexes =self.proxy_faiss_index.search(self.proxybank[rand_index][0].unsqueeze(0), batch_size)       
        return indexes[0]

    def remove_places(self, list_index):
        self.proxy_faiss_index.remove_ids(list_index)
    
    #questi due metodi si potrebbero togliere:
    #e si potrebbe togliere il "self" da proxies, proxybank e places, che di fatto vengono usati solo per inizializzare bene l'indice
    def __getself__(self):
        return self.proxies#occhio che non viene modificato quando rimuovo dei vettori dall'indice

    def __len__(self):
        return len(self.proxybank)
   

class LightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, num_classes, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True, loss_name = "contrastive_loss", miner_name = None, opt_name = "SGD", agg_arch='gem', agg_config={}):
        super().__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        self.embedding_size = descriptors_dim
        #save loss name and miner name
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.opt_name = opt_name
        # Save the aggregator name
        self.agg_arch = agg_arch
        self.agg_config = agg_config
        #save number of classes for the loss functions
        self.num_classes = num_classes
        #save embedding_size
        self.embedding_size = descriptors_dim
        #print(self.num_classes)
        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        #create the proxy head
        self.proxyhead=ProxyHead()
        # Save in_features of model.fc
        self.in_feats = self.model.fc.in_features
        # eliminate last two layers
        self.layers = list(self.model.children())[:-2]
        # define backbone
        self.backbone = torch.nn.Sequential(*self.layers)
        #the backbone outputs descriptors of dimension (num_batches = 256, 512, 7, 7)
        if self.agg_arch == "gem":
            self.aggregator = nn.Sequential(
                ag.L2Norm(),
                ag.get_aggregator(agg_arch, agg_config),
                ag.Flatten(),
                nn.Linear(self.in_feats, descriptors_dim),
                ag.L2Norm()
            )
            #after: we are going to obtain an output of size (256, 512)
        elif self.agg_arch == "mixvpr":
            self.aggregator = nn.Sequential(
                ag.get_aggregator(agg_arch, agg_config),
                nn.Linear(2048, descriptors_dim)
            )
            #self.aggregator = ag.get_aggregator(agg_arch, agg_config)
        # Set the loss function
        self.loss_fn = lm.get_loss(loss_name, num_classes, self.embedding_size)#add num_classes -> idea: send not only the name of the loss you want
                                            # but also the num_classes in case it is CosFace or ArcFace
        # Set the miner
        self.miner = lm.get_miner(miner_name)


    def forward(self, images):
        descriptors = self.backbone(images)
        #output: (256, 512, 7, 7)
        descriptors1 = self.aggregator(descriptors)
        #output: (256, 512) if gem OR (256, 2048) if mixvpr        
        descriptors2 = self.proxyhead(descriptors1)#la proxyhead va applicata dopo l'aggregator, per un'ulteriore 
        #dimensionality reduction,
        return descriptors, descriptors2

    def configure_optimizers(self):
        if self.opt_name.lower() == "sgd":
            optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        if self.opt_name.lower() == "adamw":
            optimizers = torch.optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        if self.opt_name.lower() == "adam":
            optimizers = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        if self.opt_name.lower() == "asgd":
            optimizers = torch.optim.ASGD(self.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        #cosface and arcface assume normalization ---> similar to linear layers
        if self.loss_name == "cosface" or self.loss_name == "arcface":
            self.loss_optimizer = torch.optim.SGD(self.loss_fn.parameters(), lr = 0.01)
            return [optimizers, self.loss_optimizer]
        return optimizers


    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        if self.miner is not None: #if I have selected a miner
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
        else:
            loss = self.loss_fn(descriptors, labels)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx, optimizer_idx = None): #optimizer_idx
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        # Feed forward the batch to the model
        descriptors, compact = self(images)  # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above

        #at each training iterations the compact descriptors obtained by the forward method 
        # after passing through the proxyhead are added to the bank
        global bank
        bank.adddata(compact,labels)
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors, _ = self(images) #in the inference I don't care for the descriptors of the PROXYHEAD
        return descriptors.cpu().numpy().astype(np.float32)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset)

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, num_preds_to_save=0):
        """all_descriptors contains database then queries descriptors"""
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num : ]
        database_descriptors = all_descriptors[ : inference_dataset.database_num]

        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            trainer.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        print(recalls_str)
        self.log('R@1', recalls[0], prog_bar=False, logger=True)
        self.log('R@5', recalls[1], prog_bar=False, logger=True)
        #Alla fine di ogni epoch (quando questo metodo viene chiamato), inizializzo la nuova banca
        global bank
        bank=ProxyBank()

def get_datasets_and_dataloaders(args):
    train_transform = tfm.Compose([
        tfm.RandAugment(num_ops=3),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = TrainDataset(
        dataset_folder=args.train_path,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        transform=train_transform
    )
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, batch_sampler = ProxySampler(train_dataset, args.batch_size))#BatchSampler=ProxySamplerVersione2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = parser1.parse_arguments()

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args)
    num_classes = train_dataset.__len__()
    model = LightningModel(val_dataset, test_dataset, num_classes, args.descriptors_dim, args.num_preds_to_save, args.save_only_wrong_preds, args.loss_func, args.miner, args.optimizer, args.aggr)
    
    # Model params saving using Pytorch Lightning. Save the best 3 models according to Recall@1
    checkpoint_cb = ModelCheckpoint(
        monitor='R@1',
        filename='_epoch({epoch:02d})_step({step:04d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max'
    )

    # Instantiate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./LOGS',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    """if(args.ckpt_path == None):
        trainer.validate(model=model, dataloaders=val_loader)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.ckpt_path)
    """

    if(args.only_test == False):
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path = args.ckpt_path)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path = args.ckpt_path)
        trainer.test(model = model, dataloaders=test_loader)
    else:
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.ckpt_path)