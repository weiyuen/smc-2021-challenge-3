import argparse
import cv2
import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from class_mapping import class_to_color
from transunet import TransUNet


class TestDataset(Dataset):
    '''Creates a dataset of normalized and resized images.'''
    def __init__(self, source_dir):
        self.w = self.h = 224
        self.source_dir = source_dir
        self.filenames = os.listdir(source_dir)
        self.filenames = [os.path.join(source_dir, file) for file in self.filenames]
        print(f"Found {len(self.filenames)} files.")
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, i):
        image = cv2.imread(self.filenames[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.w, self.h))
        image = image / 255.
        image = np.transpose(image, (2, 0, 1))
        image = torch.FloatTensor(image)
        return image, self.filenames[i]
    
    
def run_segmentation(model, source_dir, target_dir, batch_size):
    '''Runs model on images in source_dir and saves them in target_dir.'''
    
    # Create Dataset and DataLoader from source_dir.
    ds = TestDataset(source_dir)
    datagen = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for (images, filenames) in tqdm(datagen):
        outputs = model.forward(images.to(device))
        outputs = outputs.cpu().detach().numpy()
        outputs = np.argmax(outputs, axis=1)
        outputs = [class_to_color(seg) for seg in outputs]
        outputs = [cv2.cvtColor(seg, cv2.COLOR_RGB2BGR) for seg in outputs]
        outputs = [cv2.resize(seg, (1280, 720), interpolation=cv2.INTER_NEAREST) for seg in outputs]
        
        # Extract filename of original image to use as segmentation filename.
        filenames = [filename.split('\\')[-1] for filename in filenames]
        filenames = [filename.split('.')[0] for filename in filenames]
        filenames = [(filename + '.png') for filename in filenames]

        for seg, fn in zip(outputs, filenames):
            seg_path = os.path.join(target_dir, fn)
            cv2.imwrite(seg_path, seg)
       
    print('Segmentation complete!')


checkpoint_path = r'..\model\final_solution_checkpoint\epoch=136-step=45620.ckpt'
hparams_path = r'..\model\final_solution_checkpoint\hparams.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, help='Path to image source folder.')
parser.add_argument('--target_dir', type=str, help='Path to target folder.')
parser.add_argument('--batch_size', type=int, default=8)

args = parser.parse_args()

model = TransUNet.load_from_checkpoint(checkpoint_path, hparams=hparams_path)
model.eval();

run_segmentation(model, args.source_dir, args.target_dir, args.batch_size)