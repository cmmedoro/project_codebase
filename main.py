
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

# class to implement the GeM pooling layer, to substitute to the current Average Pooling layer of ResNet-18
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


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
        print(self.num_classes)
        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Save in_features of model.fc
        self.in_feats = self.model.fc.in_features
        # eliminate last two layers
        self.layers = list(self.model.children())[:-2]
        # define backbone
        self.backbone = torch.nn.Sequential(*self.layers)
        if self.agg_arch == "gem":
            self.aggregator = nn.Sequential(
                ag.L2Norm(),
                ag.get_aggregator(agg_arch, agg_config),
                ag.Flatten(),
                nn.Linear(self.in_feats, descriptors_dim),
                ag.L2Norm()
            )
        elif self.agg_arch == "mixvpr":
            self.aggregator = ag.get_aggregator(agg_arch, agg_config)
            self.embedding_size = 4096
        # Set the loss function
        self.loss_fn = lm.get_loss(loss_name, num_classes, self.embedding_size)#add num_classes -> idea: send not only the name of the loss you want
                                            # but also the num_classes in case it is CosFace or ArcFace
        # Set the miner
        self.miner = lm.get_miner(miner_name)


    def forward(self, images):
        descriptors = self.backbone(images)
        print("descriptors")
        print(descriptors.shape)
        descriptors = self.aggregator(descriptors)
        print(descriptors.shape)
        return descriptors

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
        descriptors = self(images)  # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)

        """if isinstance(self.optimizers, torch.optim.Optimizer):
            optimizer = self.optimizers
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer = self.optimizers[optimizer_idx]
            optimizer.step()
            optimizer.zero_grad()"""
        return {'loss': loss}

    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors = self(images)
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
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

    if(args.ckpt_path == None):
        trainer.validate(model=model, dataloaders=val_loader)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.ckpt_path)

