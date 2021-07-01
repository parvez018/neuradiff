import torch
import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader
import argparse
import unet

import math
import os

class Param:
	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 4321

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

param = Param()


# Nasim
param.data_path = '../data/irradiation_v4'
param.filename_pkl = 'all_data_train.pkl'
param.video_pkl = 'synthesized_eta_train.pkl'


param.batch_size = 64
param.lr = 1e-1
param.epoch = 20

param.embedding_features = 8

param.best_unet_model_out = 'best_video_train_only_unet_ef8.model'


class IrradiationVideoDataset(Dataset):
	def __init__(self, data_path, filename_pkl, video_pkl, skip_step=1):
		super(IrradiationVideoDataset, self).__init__()
		self.all_data = torch.load(os.path.join(data_path, filename_pkl))
		self.video_data = torch.load(os.path.join(data_path, video_pkl))
		self.video_data = self.video_data.type(torch.float)
		self.video_data -= 128.0
		self.video_data /= 128.0
		self.start_skip = 9
		self.skip_step = skip_step
		self.cnt = len(self.all_data) -self.start_skip*2 -skip_step

		prob = torch.ones(self.cnt + skip_step)*.5
		self.upper_lower = torch.bernoulli(prob)

	def __getitem__(self, index):
		return {'cv': self.all_data[index+self.start_skip]['cv'][1:-1,1:-1],
				'ci': self.all_data[index+self.start_skip]['ci'][1:-1,1:-1],
				'eta': self.all_data[index+self.start_skip]['eta'][1:-1,1:-1],
				'cv_ref': self.all_data[index+self.start_skip+self.skip_step]['cv'][1:-1,1:-1],
				'ci_ref': self.all_data[index+self.start_skip+self.skip_step]['ci'][1:-1,1:-1],
				'eta_ref': self.all_data[index+self.start_skip+self.skip_step]['eta'][1:-1,1:-1],
				'v': self.video_data[index,:,:,:],
				'v_ref': self.video_data[index+self.skip_step,:,:,:],
				'index':index,
				'index_ref':index+self.skip_step,
				'ul': self.upper_lower[index],
				'ul_ref': self.upper_lower[index+self.skip_step]
		}

	def __len__(self):
		return self.cnt


def stitch_by_mask(cv, cv_ori, mask, ul1):
	ul = ul1.unsqueeze(-1).unsqueeze(-1)
	ul = ul.repeat(1, 128, 128)
	mask1 = mask*ul + (1-mask)*(1-ul)
	return cv_ori * mask1 + cv * (1 - mask1)


def main_v0():
		# initialize
		print('parameters: ', param)

		skip_step = 2
		dataset = IrradiationVideoDataset(param.data_path, param.filename_pkl, param.video_pkl, skip_step)
		loader = DataLoader(dataset, batch_size=param.batch_size, shuffle=True)
		mse = nn.MSELoss(reduction='sum').to(device)

		minloss = 100000000.0

		video2pf = unet.UNetWithEmbeddingGen(in_channels=3, out_channels=3,
											 init_features=32,
											 embedding_features=param.embedding_features,
											 video_len=dataset.__len__() + skip_step,
		                                     fix_emb=False)
		video2pf = video2pf.to(device)

		optimizer = torch.optim.Adam(video2pf.parameters(), lr=param.lr)

		mask = torch.ones((128, 128), dtype=torch.float).to(device)
		mask[64:128, :] = 0.0

		frame0 = dataset.__getitem__(0)
		ci0 = frame0['ci'].to(device).unsqueeze_(dim=0)
		ci0 = ci0.repeat(param.batch_size, 1, 1)
		print('ci0.size()=', ci0.size())

		for i in range(param.epoch):
				loss = 0.0
				total_size = 0
				print('epoch: ', i)
				for batch in loader:
						cv1 = batch['cv'].to(device)
						ci1 = batch['ci'].to(device)
						eta1 = batch['eta'].to(device)
						frame1 = batch['v'].to(device)
						indicies1 = batch['index'].to(device)

						# get cv, ci, eta from the movie frame
						pf = video2pf(frame1, indicies1)

						cv = pf[:,0,:,:]
						ci = pf[:,1,:,:]
						eta = pf[:,2,:,:]

						# print('eta1.size()=', eta1.size())
						# print('eta.size()=', eta.size())
						# print('cv.size()=', cv.size())
						# print('ci.size()=', ci.size())
						# print('ci0.size()=', ci0.size())

						this_batch_size = ci.size()[0]
						if this_batch_size < param.batch_size:
							ci00 = ci0[0:this_batch_size,:,:]
						else:
							ci00 = ci0

						ul1 = batch['ul'].to(device)
						ul1.unsqueeze_(-1).unsqueeze_(-1)
						ul1 = ul1.repeat(1, 128, 128)
						mask1 = mask * ul1 + (1 - mask) * (1 - ul1)

						batch_loss = mse(mask1*eta1, mask1*eta) + \
									 mse(mask1*eta1, mask1*cv) + \
									 mse(mask1*ci00, mask1*ci)

						optimizer.zero_grad()
						batch_loss.backward()
						optimizer.step()

						this_size = cv1.size(0)
						loss += batch_loss.item()
						total_size += this_size

						print('loss: %.8f' % (loss / total_size))


				loss /= total_size
				print('loss total: %.8f' % loss)
				if loss < minloss:
						minloss = loss
						print('Get minimal loss')

						torch.save(video2pf.state_dict(), param.best_unet_model_out)

				print('')

if __name__ == '__main__':
		main_v0()
