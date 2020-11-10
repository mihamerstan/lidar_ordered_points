import argparse
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from utils import data
import models, utils

import pandas as pd
from laspy.file import File
from pickle import dump, load

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as udata
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

import lidar_data_processing


# Functions
def add_mask(sample,mask_pts_per_seq,consecutive=True):
    # Given a 3-D tensor of all ones, returns a mask_tensor of same shape 
    # with random masking determined by mask_pts_per_seq
    mask_tensor = torch.ones(sample.shape)
    seq_len = mask_tensor.shape[0]
    
    if consecutive:
        # Creates a square of missing points
        first_mask = int(np.random.choice(np.arange(8,seq_len-8-mask_pts_per_seq),1))
        
        mask_tensor[first_mask:first_mask+mask_pts_per_seq,first_mask:first_mask+mask_pts_per_seq,:] = 0
            
    else:
        # TO DO: Random points throughout the patch
        for i in range(sample.shape[0]):
            m[i,:] = np.random.choice(np.arange(8,seq_len-8),mask_pts_per_seq,replace=False)

    return mask_tensor

# Loss function that weights the loss according to coordinate ranges (xmax-xmin, ymax-ymin, zmax-zmin)
def weighted_MSELoss(pred,true,sc,mask_pts_per_seq=5):
    '''weighted_MSELoss reconverts MSE loss back to the original scale of x,y,z.
    Rationale is because xyz have such different ranges, we don't want to ignore the ones with largest scale.
    Assumes that x,y,z are the first 3 features in sc scaler'''
    
    ranges = torch.Tensor(sc.data_max_[:3]-sc.data_min_[:3])
    raw_loss = torch.zeros(3,dtype=float)
    for i in range(3):
        raw_loss[i] = F.mse_loss(pred[:,i,:,:], true[:,i,:,:], reduction="sum") 
    return (ranges**2 * raw_loss).sum() #/ (pred.shape[0]*mask_pts_per_seq**2)

# Dataloader class
class LidarLstmDataset(udata.Dataset):
    def __init__(self, scan_line_tensor, idx_list, seq_len = 64, mask_pts_per_seq = 5, consecutive = True):
        super(LidarLstmDataset, self).__init__()
        self.scan_line_tensor = scan_line_tensor
        self.idx_list = idx_list
        self.seq_len = seq_len
        self.mask_pts_per_seq = mask_pts_per_seq
        self.consecutive = consecutive

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self,index):
        row = self.idx_list[index][0]
        col = self.idx_list[index][1]
        clean = self.scan_line_tensor[row:row+self.seq_len,col:col+self.seq_len,:]
        mask = add_mask(clean,self.mask_pts_per_seq,self.consecutive)
        return clean.permute(2,0,1), mask.permute(2,0,1)



# Args


def main(args):
	# gpu or cpu
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	args = utils.setup_experiment(args)
	utils.init_logging(args)


	# Loading models
	MODEL_PATH_LOAD = "../lidar_experiments/2d/lidar_unet2d/lidar-unet2d-Nov-08-16:29:49/checkpoints/checkpoint_best.pt"

	train_new_model = True

	# Build data loaders, a model and an optimizer
	if train_new_model:
	    model = models.build_model(args).to(device)
	else:
	    model = models.build_model(args)
	    model.load_state_dict(torch.load(args.MODEL_PATH_LOAD)['model'][0])
	    model.to(device)

	print(model)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,15,30,50,100,250], gamma=0.5)
	logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

	if args.resume_training:
	    state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
	    global_step = state_dict['last_step']
	    start_epoch = int(state_dict['last_step']/(403200/state_dict['args'].batch_size))+1
	else:
	    global_step = -1
	    start_epoch = 0


	## Load the pts files
	# Loads as a list of numpy arrays
	scan_line_tensor = torch.load(args.data_path+'scan_line_tensor.pts')
	train_idx_list = torch.load(args.data_path+'train_idx_list.pts')
	valid_idx_list = torch.load(args.data_path+'valid_idx_list.pts')
	sc = torch.load(args.data_path+'sc.pts')

	# Dataloaders
	train_dataset = LidarLstmDataset(scan_line_tensor,train_idx_list,args.seq_len, args.mask_pts_per_seq)
	valid_dataset = LidarLstmDataset(scan_line_tensor,valid_idx_list,args.seq_len, args.mask_pts_per_seq)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, num_workers=4, shuffle=True)

	# Track moving average of loss values
	train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss"])}
	valid_meters = {name: utils.AverageMeter() for name in (["valid_loss"])}
	writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

	##################################################
	# TRAINING
	for epoch in range(start_epoch, args.num_epochs):
	    if args.resume_training:
	        if epoch %1 == 0:
	            optimizer.param_groups[0]["lr"] /= 2
	            print('learning rate reduced by factor of 2')

	    train_bar = utils.ProgressBar(train_loader, epoch)
	    for meter in train_meters.values():
	        meter.reset()

	#     epoch_loss_sum = 0
	    for batch_id, (clean, mask) in enumerate(train_bar):
	        # dataloader returns [clean, mask] list
	        model.train()
	        global_step += 1
	        inputs = clean.to(device)
	        mask_inputs = mask.to(device)
	        # only use the mask part of the outputs
	        raw_outputs = model(inputs,mask_inputs)
	        outputs = (1-mask_inputs[:,:3,:,:])*raw_outputs + mask_inputs[:,:3,:,:]*inputs[:,:3,:,:]
	        
	        if args.wtd_loss:
	            loss = weighted_MSELoss(outputs,inputs[:,:3,:,:],sc)/(inputs.size(0)*(args.mask_pts_per_seq**2))
	            # Regularization?
	            
	        else:
	            # normalized by the number of masked points
	            loss = F.mse_loss(outputs, inputs[:,:3,:,:], reduction="sum") / \
	                   (inputs.size(0) * (args.mask_pts_per_seq**2))

	        model.zero_grad()
	        loss.backward()
	        optimizer.step()
	#         epoch_loss_sum += loss * inputs.size(0)
	        train_meters["train_loss"].update(loss)
	        train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

	        if writer is not None and global_step % args.log_interval == 0:
	            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
	            writer.add_scalar("loss/train", loss.item(), global_step)
	            gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
	            writer.add_histogram("gradients", gradients, global_step)
	            sys.stdout.flush()
	#     epoch_loss = epoch_loss_sum / len(train_loader.dataset)
	    
	    if epoch % args.valid_interval == 0:
	        model.eval()
	        for meter in valid_meters.values():
	            meter.reset()

	        valid_bar = utils.ProgressBar(valid_loader)
	        val_loss = 0
	        for sample_id, (clean, mask) in enumerate(valid_bar):
	            with torch.no_grad():
	                inputs = clean.to(device)
	                mask_inputs = mask.to(device)
	                # only use the mask part of the outputs
	                raw_output = model(inputs,mask_inputs)
	                output = (1-mask_inputs[:,:3,:,:])*raw_output + mask_inputs[:,:3,:,:]*inputs[:,:3,:,:]

	                # TO DO, only run loss on masked part of output
	                
	                if args.wtd_loss:
	                    val_loss = weighted_MSELoss(output,inputs[:,:3,:,:],sc)/(inputs.size(0)*(args.mask_pts_per_seq**2))
	                else:
	                    # normalized by the number of masked points
	                    val_loss = F.mse_loss(output, inputs[:,:3,:,:], reduction="sum")/(inputs.size(0)* \
	                                                                                    (args.mask_pts_per_seq**2))

	                valid_meters["valid_loss"].update(val_loss.item())

	        if writer is not None:
	            writer.add_scalar("loss/valid", valid_meters['valid_loss'].avg, global_step)
	            sys.stdout.flush()

	        logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=optimizer.param_groups[0]["lr"])))
	        utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")
	    scheduler.step()

	logging.info(f"Done training! Best Loss {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")



def get_args():
	parser = argparse.ArgumentParser(allow_abbrev=False)

	# Add data arguments
	parser.add_argument("--data-path", default="../lidar_data/32_32/", help="path to data directory")
	parser.add_argument("--dataset", default="masked_pwc", help="masked training data for generator")
	parser.add_argument("--batch-size", default=64, type=int, help="train batch size")
	parser.add_argument("--num_scan_lines", default=1000, type=int, help="number of scan lines used to generate data")
	parser.add_argument("--seq_len", default=32, type=int, help="side length of the patches")
	parser.add_argument("--scan_line_gap_break", default=7000, type=int, help="threshold over which scan_gap indicates a new scan line")
	parser.add_argument("--min_pt_count", default=1700, type=int, help="in a scan line, otherwise line not used")
	parser.add_argument("--max_pt_count", default=2000, type=int, help="in a scan line, otherwise line not used")
	parser.add_argument("--mask_pts_per_seq", default=5, type=int, help="Sqrt(masked pts), side of the missing patch")
	parser.add_argument("--mask_consecutive", default=True, help="True if pts are in a consecutive patch")
	parser.add_argument("--stride_inline", default=5, type=int, help="The number of pts skipped between patches within the scan line")
	parser.add_argument("--stride_across_lines", default=3, type=int, help="The number of pts skipped between patches across the scan line")

	# parser.add_argument("--n-data", default=100000,type=int, help="number of samples")
	# parser.add_argument("--min_sep", default=5,type=int, help="minimum constant sample count for piecwewise function")


	# Add model arguments
	parser.add_argument("--model", default="lidar_unet2d", help="Model used")
	# parser.add_argument("--in_channels", default=7, type=int, help="Number of in channels")
	# parser.add_argument("--modelG", default="unet1d", help="Generator model architecture")
	# parser.add_argument("--modelD", default="gan_discriminator", help="Discriminator model architecture")
	parser.add_argument("--wtd_loss", default=True, help="True if MSELoss should be weighted by xyz distances")
	# parser.add_argument("--g_d_update_ratio", default = 2, type=int, help="How many times to update G for each update of D")

	# Add optimization arguments
	parser.add_argument("--lr", default=.005, type=float, help="learning rate for generator")
	parser.add_argument("--weight_decay", default=0., type=float, help="weight decay for optimizer")

	# Logistics arguments
	parser.add_argument("--num-epochs", default=10, type=int, help="force stop training at specified epoch")
	parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
	parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")
	parser.add_argument("--output_dir", default='../lidar_experiments/2d', help="where the model and logs are saved.")	
	parser.add_argument("--MODEL_PATH_LOAD", default='../lidar_experiments/2d/lidar_unet2d/lidar-unet2d-Nov-08-16:29:49/checkpoints/checkpoint_best.pt', help="where to load an existing model from")	
	# Parse twice as model arguments are not known the first time
	parser = utils.add_logging_arguments(parser)
	args, _ = parser.parse_known_args()
	models.MODEL_REGISTRY[args.model].add_args(parser)
	# models.MODEL_REGISTRY[args.modelD].add_args(parser)
	args = parser.parse_args()
	print("vars(args)",vars())
	return args


if __name__ == "__main__":
	args = get_args()
	main(args)
