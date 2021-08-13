import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS
from utils import DiceLoss


class TransUNet(pl.LightningModule):
    
    def __init__(self, vit_config, args, lr=0.0001):
        super().__init__()
        if args['vit_name'].find('R50') != -1:
            vit_config.patches.grid = (int(args['img_size']/args['vit_patches_size']),
                                       int(args['img_size']/args['vit_patches_size']))
            
        self.vitseg = ViT_seg(vit_config, img_size=args['img_size'], num_classes=args['n_classes'])
        self.vitseg.load_from(weights=np.load(vit_config.pretrained_path))
        self.vit_config = vit_config
        self.args = args
        self.lr = lr
        
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(args['n_classes'])
        
    def forward(self, x):
        x = self.vitseg(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        
        # Loss = CE + DICE
        loss_ce = self.ce_loss(outputs, y[:].long())
        loss_dice = self.dice_loss(outputs, y, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        
        loss_ce = self.ce_loss(outputs, y[:].long())
        loss_dice = self.dice_loss(outputs, y, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        outputs = self.forward(x)
        
        loss_ce = self.ce_loss(outputs, y[:].long())
        loss_dice = self.dice_loss(outputs, y, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('test_loss', loss)
        return loss
        
    def configure_optimizers(self):
        base_lr = self.args['base_lr']
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        return optimizer   