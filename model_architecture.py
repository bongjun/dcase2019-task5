import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()

		self.layer1_conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
		self.layer2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

									
		self.layer3_conv2 = nn.Sequential(nn.Conv2d(64, 128,kernel_size=3, stride=1, padding=1), nn.ReLU())
		self.layer4_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.layer5_conv3_1 = nn.Sequential(nn.Conv2d(128, 256,kernel_size=3, stride=1,padding=1), nn.ReLU())
		self.layer6_conv3_2 = nn.Sequential(nn.Conv2d(256, 256,kernel_size=3, stride=1,padding=1), nn.ReLU())
		self.layer7_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.layer8_conv4_1 = nn.Sequential(nn.Conv2d(256, 512,kernel_size=3, stride=1,padding=1), nn.ReLU())
		self.layer9_conv4_2 = nn.Sequential(nn.Conv2d(512, 512,kernel_size=3, stride=1,padding=1), nn.ReLU())
		
		self.new_fc1 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU())
		self.new_fc2 = nn.Sequential(nn.Linear(2048, 128), nn.ReLU())

		self.final= nn.Sequential(nn.Linear(128, 8), nn.Sigmoid())

	
	def forward(self, x):

		out = self.layer1_conv1(x)
		out = self.layer2_pool1(out)

		out = self.layer3_conv2(out)
		out = self.layer4_pool2(out)

		out = self.layer5_conv3_1(out)
		out = self.layer6_conv3_2(out)
		out = self.layer7_pool3(out)

		out = self.layer8_conv4_1(out)
		out = self.layer9_conv4_2(out)

		# maxpooling
		out = torch.max(out, dim=2)[0]
		out = out.view(out.size(0),-1)
	
		out = self.new_fc1(out)
		out = self.new_fc2(out)
		out = self.final(out)
	
		return out