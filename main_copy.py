
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
        #self.proxies= Proxies(bank)
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
    def __init__(self, dataset, batch_size, bank, generator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = len(self.dataset)//self.batch_size 
        self.generator = generator
        self.bank = bank
        self.first_epoch = 0
        #Take the bank you have defined at the end of the previous epoch (in inference epoch end)
        #compute the final averages and instantiate the index
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator() # to produce pseudo random numbers
        self.generator.manual_seed(seed)
        # counter for number of times __iter__ is called
        self.itercounter = 0
        self.batches = []
        
    def __iter__(self): # Lightening Module calls it twice for each epoch
        if self.first_epoch==0 and self.itercounter % 2 == 0:
            self.first_epoch=1
            #index_bank=list(range(len(self.dataset)))
            """
            while len(index_bank)>self.batch_size:
                indexes=random.sample(index_bank, self.batch_size)
                batches.append(indexes)
                #for el in indexes:
                #    index_bank.pop(el)
                #batches.append(torch.randperm(self.batch_size, generator=self.generator).tolist())
            batches.append(indexes)"""
            # torch.randperm(n) returns a random permutation of numbers from 0 to n-1
            # generator = pseudo-random number generator for sampling
            # numbers from 0 to len(dataset) ---> split the returned list into a number of sublists = batch_size
            self.batches = torch.randperm(len(self.dataset), generator=self.generator).split(self.batch_size)
            #print("Shape of batches at first epoch")
            #print(len(self.batches))
            #self.itercounter += 1
            #return iter(self.batches)
        elif self.itercounter % 2 == 0:
            #print("Casini nel random evitati")
            #print(self.bank.__len__())
            #print("Number of keys in the bank")
            #print(len(self.bank.getkeys()))
            self.bank.computeavg()
            self.bank.update_index()
            self.batches=[]
            while self.bank.__len__() > self.batch_size:
                randint = random.choice(self.bank.getkeys()) # selects randomly an element among the keys (places) of the bank
                # take neareast neighbors of the random place as selected places for the new batch
                indexes = self.bank.getproxies(rand_index=randint, batch_size=self.batch_size)
                # then remove selected places both from bank and from index
                self.bank.remove_places(indexes)
                self.batches.append(indexes.tolist())
            self.batches.append(self.bank.getkeys())  
        self.bank.reset()
        self.itercounter += 1
        #return iter(batches)
        return iter(self.batches)
        """Sampler used as model:
        combined = list(first_half_batches + second_half_batches)
        combined = [batch.tolist() for batch in combined]
        random.shuffle(combined)
        return iter(combined)"""
            
    def __len__(self):
        return self.length

class ProxyHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #define a dimensionality reduction operation considering (512, 256)
        self.dimred= nn.Linear(in_channels, out_channels) 
        self.norm=ag.L2Norm()#Ragionare bene su quale dimensione devo andare ad agire
        #di default la dimensione è la 1
        #dovrei ricevere in input qualcosa che ha come dimensione[0] 256 (le immagini di un batch)
        #e come dimensione[1] 512, ovvero descriptors_dim

    def forward(self, x):
        #apply dimensionality reduction
        x = self.dimred(x)
        #apply normalization
        x = self.norm(x)
        return x
    

class ProxyBank():
    # Dictionary: to each k (place index) we associate a tensor as the compact descriptor of that place, obtained as the mean of the
    # feature maps obtained from images of that place, rescaled
    def __init__(self, descriptor_dimension):
        self.dim = descriptor_dimension
        # initialize the proxy bank as an empty dictionary
        self.proxybank = {}
        # define IndexFlatL2: measures the L2 distance between all given points of the query vector
        # and the vectors loaded into the index
        # initialize the index with the vector dimensionality = descriptors_dim
        support_index = faiss.IndexFlatL2(self.dim)
        # in order to be able to translate ids when mapping and searching we need to use IndexIDMap to encapsulate the previously 
        # created index ---> needed because faiss proposes sequential ids, instead we would want to specify our own ids.
        self.proxy_faiss_index = faiss.IndexIDMap(support_index)

    def adddata(self, compact_descriptors, labels):
        # at each batch of the neural network we need to add the new descriptors
        for compact_descriptor, label in zip(compact_descriptors, labels):
            label = int(label)
            if label in self.proxybank.keys(): #if the label is already among the keys of the proxy bank, I just need to update the values
                #old = self.proxybank[label][0]
                #old_count = self.proxies[label][1]
                #old += compact_descriptor
                #old_count += 1
                self.proxybank[label][0] = self.proxybank[label][0] + compact_descriptor
                self.proxybank[label][1] = self.proxybank[label][1] + 1
            else: 
                # create a new entry of the kind [compact_descriptor, counter_to_increment] for the new label
                self.proxybank[label] = [compact_descriptor, 1]
    
    def computeavg(self):
        # at the end of each epoch we need to compute the compact descriptors for each place
        for el in self.proxybank.values():
            # el[0] ---> compact_descriptor
            # el[1] ---> counter with number of descriptors for that label
            el[0] = el[0]/el[1]

    def update_index(self):
        # save the places as the list of the keys of the proxy bank
        self.places = np.array(list(self.proxybank.keys())) # after initialization it is not modified
        #print("Number of places when updating index")
        #print(len(self.places))
        # define the proxies ---> for each place in self.places, consider the compact descriptor in the bank corresponding to
        # that place. Create an array
        self.proxies = np.array([self.proxybank[key][0].detach().cpu().numpy() for key in self.places]) #.numpy().astype(np.float32)
        #print("Shape of proxies when updating index")
        #print(self.proxies.shape)
        # add the proxies and the places (labels) to the index
        self.proxy_faiss_index.add_with_ids(self.proxies, self.places)
    
    def getproxies(self, rand_index, batch_size):
        # Here you want to get the k = batch_size closest descriptors to the one corresponding to the rand_index
        _, indexes = self.proxy_faiss_index.search(self.proxybank[rand_index][0].unsqueeze(0).detach().cpu().numpy(), batch_size)       
        #alternativa: self.proxy_faiss_index.search(self.proxies[rand_index], batch_size)
        #ma ti devi fidare di come lui mette i descrittori dentro l'indice
        return indexes[0]

    def reset(self):
      # reset the values for the bank as in __init__()
      self.proxybank = {}
      support_index = faiss.IndexFlatL2(self.dim)
      self.proxy_faiss_index = faiss.IndexIDMap(support_index)
    
    def getdict(self):
        return self.proxybank
        
    def getkeys(self):
        return list(self.proxybank.keys())
    
    def remove_places(self, list_index):
      # remove elements from the proxy bank and from the index
      for el in list_index:
        self.proxybank.pop(el)
        self.proxy_faiss_index.remove_ids(list_index)

    def __getitem__(self, key):
        return self.proxybank[key][0]
    
    def __len__(self):
        return len(self.proxybank)

class LightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, num_classes, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True, sched_name = None, max_epochs = 20, loss_name = "contrastive_loss", miner_name = None, opt_name = "SGD", agg_arch='gem', bank=None, agg_config={}):
        super().__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        # add num_classes
        self.num_classes = num_classes
        #print(num_classes)
        #print(self.num_classes)
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        # save loss name and miner name
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.opt_name = opt_name
        self.sched_name = sched_name
        # Save the aggregator name
        self.agg_arch = agg_arch
        self.agg_config = agg_config
        self.embedding_size = descriptors_dim
        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # create the proxy head
        self.proxyhead = ProxyHead()
        # Save in_features of model.fc
        self.in_feats = self.model.fc.in_features
        # eliminate last two layers
        self.layers = list(self.model.children())[:-2]
        # define backbone
        self.backbone = torch.nn.Sequential(*self.layers)
        # define the bank
        self.bank = bank
        # define the aggregator
        if self.agg_arch == "gem":
            self.aggregator = nn.Sequential(
                ag.L2Norm(),
                ag.get_aggregator(agg_arch, agg_config),
                ag.Flatten(),
                nn.Linear(self.in_feats, descriptors_dim),
                ag.L2Norm()
            )
        elif self.agg_arch == "mixvpr":
            self.aggregator = nn.Sequential(
                ag.get_aggregator(agg_arch, agg_config),
                nn.Linear(2048, descriptors_dim)
            )
        # Set the loss function
        self.loss_fn = lm.get_loss(loss_name, num_classes, self.embedding_size)#idea: send not only the name of the loss you want
                                            # but also the num_classes in case it is CosFace or ArcFace
        # Define the loss for the proxy head
        self.loss_head = lm.get_loss(loss_name, num_classes, self.embedding_size)
        # Set the miner
        self.miner = lm.get_miner(miner_name)
        

    def forward(self, images):
        descriptors = self.backbone(images)
        descriptors1 = self.aggregator(descriptors)
        #apply the proxyhead to obtain a new dimensionality reduction
        descriptors2 = self.proxyhead(descriptors1)
        #print("Descriptors shape (output of aggregator)")
        #print(descriptors1.shape)
        #print("Output proxy")
        #print(descriptors2.shape)
        return descriptors1, descriptors2

    def configure_optimizers(self):
        if self.opt_name.lower() == "sgd":
            optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        if self.opt_name.lower() == "adamw":
            optimizers = torch.optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        if self.opt_name.lower() == "adam":
            optimizers = torch.optim.Adam(self.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        if self.opt_name.lower() == "asgd":
            optimizers = torch.optim.ASGD(self.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        # define the scheduler to adjust the learning rate
        if(self.sched_name == None):
            scheduler = None
        elif(self.sched_name.lower() == "cosineannealing"):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, self.max_epochs)
        elif(self.sched_name.lower() == "plateau"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode = "min", patience = 2)
        elif(self.sched_name.lower() == "onecycle"):
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizers, max_lr = 0.01, epochs = self.max_epochs, steps_per_epoch = len(train_loader))
        #cosface and arcface assume normalization ---> similar to linear layers
        if self.loss_name == "cosface" or self.loss_name == "arcface":
            self.loss_optimizer = torch.optim.SGD(self.loss_fn.parameters(), lr = 0.01)
            if(scheduler is None):
                return [optimizers, self.loss_optimizer]
            #return [optimizers, self.loss_optimizer], scheduler
            return {"optimizer": [optimizers, self.loss_optimizer], "lr_scheduler": scheduler, "monitor" : "loss"}
        if(scheduler is None):
            return optimizers
        #return [optimizers], scheduler
        return {"optimizer": optimizers, "lr_scheduler": scheduler, "monitor" : "loss"}
    #{"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}


    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        if self.miner is not None: #if I have selected a miner
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
        else:
            loss = self.loss_fn(descriptors, labels)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx, optimizer_idx = None):
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        # Feed forward the batch to the model
        descriptors, compact = self(images)  # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) + self.loss_head(compact,labels) # Call the loss_function we defined above

        #at each training iterations the compact descriptors obtained by the forward method after passing through the proxyhead 
        #are added to the bank
        #print("shape of compact descriptors at training step")
        #print(compact.shape)
        self.bank.adddata(compact, labels)
        #print("length of bank after adddata in training_step")
        #print(self.bank.__len__())"""
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors, _ = self(images) #apply the model to the images to obtain the descriptors
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
        

def get_datasets_and_dataloaders(args, bank):
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
    #train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_sampler = ProxySampler(train_dataset, args.batch_size, bank))#BatchSampler=ProxySamplerVersione2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = parser1.parse_arguments()

    # define the ProxyBank
    #bank = ProxyBank(args.descriptors_dim)
    bank = ProxyBank(256)
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args, bank)
    num_classes = train_dataset.__len__()
    model = LightningModel(val_dataset, test_dataset, num_classes, args.descriptors_dim, args.num_preds_to_save, args.save_only_wrong_preds, args.scheduler, args.max_epochs, args.loss_func, args.miner, args.optimizer, args.aggr, bank = bank)
    
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

    #if(args.ckpt_path == None):
     #   trainer.validate(model=model, dataloaders=val_loader)
      #  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.ckpt_path)

    if(args.only_test == False):
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path = args.ckpt_path)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path = args.ckpt_path)
        trainer.test(model = model, dataloaders=test_loader)
    else:
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.ckpt_path)

