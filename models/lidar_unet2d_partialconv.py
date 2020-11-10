import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import register_model
from models import PartialConv2d, PartialConvTranspose2d
import math

### 1D PartialConv U-NET
@register_model("lidar_unet2d")
class UNet(nn.Module):
	"""UNet as defined in https://arxiv.org/abs/1805.07709"""
	def __init__(self, bias, in_channels=7, out_channels=3, residual_connection = False, batch_norm=False):
		super(UNet, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.batch_norm = batch_norm

		self.conv1 = PartialConv2d.PartialConv2d(self.in_channels,32,5,padding = 2, bias = bias, multi_channel=False)
		self.conv2 = PartialConv2d.PartialConv2d(32,32,3,padding = 1, bias = bias, multi_channel=False)
		self.conv3 = PartialConv2d.PartialConv2d(32,64,3,stride=2, padding = 1, bias = bias, multi_channel=False)
		self.conv4 = PartialConv2d.PartialConv2d(64,64,3,padding = 1, bias=bias, multi_channel=False)
		self.conv5 = PartialConv2d.PartialConv2d(64,64,3,dilation=2, padding = 2, bias = bias, multi_channel=False)
		self.conv6 = PartialConv2d.PartialConv2d(64,64,3,dilation = 4,padding = 4, bias = bias, multi_channel=False)
		self.conv7 = PartialConvTranspose2d.PartialConvTranspose2d(64,64, 4,stride = 2, padding = 1, bias = bias)
		self.conv8 = PartialConv2d.PartialConv2d(96,32,3,padding=1, bias = bias, multi_channel=False)
		self.conv9 = PartialConv2d.PartialConv2d(32,self.out_channels,5,padding = 2, bias = False, multi_channel=False)

		self.dropout = nn.Dropout(p=0.1)
		self.residual_connection = residual_connection;
		
		self.bn32 = nn.BatchNorm2d(32)
		self.bn64 = nn.BatchNorm2d(64)

	@staticmethod
	def add_args(parser):
		"""Add model-specific arguments to the parser."""
		# TO DO: add image_channels, other arguments
		parser.add_argument("--bias", action='store_true', help="use residual bias")
		parser.add_argument("--residual", action='store_true', help="use residual connection")
		parser.add_argument("--in_channels",default=6, help="number of input features")
		parser.add_argument("--batch_norm",default=False, help="Whether or not to use batchnorm layers")
	@classmethod
	def build_model(cls, args):
		return cls(bias=args.bias, in_channels=args.in_channels, residual_connection=args.residual)

	def forward(self, x, mask_in = None):
		# Pad both x and the mask in the last dimension
		pad_right = x.shape[-2]%2
		pad_bottom = x.shape[-1]%2
		padding = nn.ZeroPad2d((0, pad_right,  0, pad_bottom))
		x = padding(x)
		mask_in = padding(mask_in)

		# Layer by layer. ReLu's can't take the mask as input
		prelu, mask = self.conv1(x, mask_in)
		out = F.relu(prelu)
		if self.batch_norm:
			out = self.bn32(out)
		# Dropout layer
		# out = self.dropout(out)
		prelu, mask_saved = self.conv2(out, mask)
		out_saved = F.relu(prelu)
		if self.batch_norm:
			out = self.bn32(out)

		# Dropout layer
		# out_saved = self.dropout(out_saved)

		prelu, mask = self.conv3(out_saved, mask_saved)
		out = F.relu(prelu)
		if self.batch_norm:
			out = self.bn64(out)

		# Dropout layer
		# out = self.dropout(out)

		prelu, mask = self.conv4(out, mask)
		out = F.relu(prelu)
		if self.batch_norm:
			out = self.bn64(out)

		# Dropout layer
		# out = self.dropout(out)

		prelu, mask = self.conv5(out, mask)
		out = F.relu(prelu)
		if self.batch_norm:
			out = self.bn64(out)

		# Dropout layer
		# out = self.dropout(out)

		prelu, mask = self.conv6(out, mask)
		out = F.relu(prelu)
		if self.batch_norm:
			out = self.bn64(out)

		# out = F.relu(self.conv7(out)) # THE TRANSPOSE

		# Dropout layer
		# out = self.dropout(out)

		prelu,mask = self.conv7(out,mask)
		out = F.relu(prelu)
		# Add the connection from layer 2
		out = torch.cat([out,out_saved],dim = 1)
		mask = torch.cat([mask,mask_saved],dim=1)
		# Expand the mask [32,1,25] back to [32,1,50]
		# mask = mask.repeat_interleave(2,dim=2)		
		if self.batch_norm:
			out = self.bn64(out)

		# Dropout layer
		# out = self.dropout(out)

		prelu, mask = self.conv8(out,mask)
		out = F.relu(prelu)
		if self.batch_norm:
			out = self.bn32(out)

		# Dropout layer
		# out = self.dropout(out)

		prelu, mask = self.conv9(out,mask)
		out = F.relu(prelu)

		if self.residual_connection:
			out = x - out;

		if pad_bottom > 0:
			out = out[:, :, :-1]
		if pad_right > 0:
			out = out[:, :, :-1, :]
		return out
