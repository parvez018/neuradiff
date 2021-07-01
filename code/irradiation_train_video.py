import torch
import torch.nn as nn
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
from irradiation_model import IrradiationSingleTimestep

import unet

import math
import os
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 4321

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description='Train Neural-Phase Net.')
parser.add_argument("--data_path", type=str, default='../data/irradiation_v4')
parser.add_argument("--filename_pkl", type=str, default='all_data_train.pkl')
parser.add_argument("--video_pkl", type=str, default='synthesized_eta_train.pkl')


parser.add_argument("--lr", type=float, default=1e-1, help="learning rate for phase-field net.")
parser.add_argument("--lr2", type=float, default=1e-3, help="learning rate for recognition net.")
parser.add_argument("--lambda1", type=float, default=10.0)
parser.add_argument("--lambda2", type=float, default=10.0)
parser.add_argument("--skip_step", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epoch", type=int, default=500)
parser.add_argument("--embedding_features", type=int, default=8)

parser.add_argument("--translation_method", type=str, default="unmt", choices=['identity', 'noisy_cloze', 'unmt'],
                        help="define the method to generate clozes -- either the Unsupervised NMT method (unmt),"
                             " or the identity  or noisy cloze baseline methods. UNMT is recommended for downstream performance, "
                             " but the noisy_cloze is relatively stong on downstream QA and fast to generate. Default is unmt"
                        )
parser.add_argument("--use_named_entity_clozes", action='store_true',
                        help="pass this flag to use named entity answer prior instead of noun phrases (recommended for downstream performance) ")
parser.add_argument('--use_subclause_clozes', action='store_true',
                        help="pass this flag to shorten clozes with constituency parsing instead of using sentence boundaries (recommended for downstream performance)")
parser.add_argument('--use_wh_heuristic', action='store_true',
                        help="pass this flag to use the wh-word heuristic (recommended for downstream performance). Only compatable with named entity clozes")
args = parser.parse_args()

data_path = args.data_path
filename_pkl = args.filename_pkl
video_pkl = args.video_pkl

best_ts_model_out = 'best_ts_model_out.model'
best_unet_model_out = 'best_unet_model_out.model'
best_unet_model_pretrained = 'best_video_train_only_unet_ef8.model'

best_ts_model_pretrained = None

lr = args.lr
lr2 = args.lr2
lambda1 = args.lambda1
lambda2 = args.lambda2
skip_step = args.skip_step
batch_size = args.batch_size
epoch = args.epoch

embedding_features = 8

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

	def __getitem__(self, idx):
		index = self.all_data[idx]['step']
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
		dataset = IrradiationVideoDataset(data_path,
										  filename_pkl,
										  video_pkl,
										  skip_step)
		# initialize
		ts_model = IrradiationSingleTimestep()
		new_record = False
		ts_model = ts_model.to(device)
		if best_ts_model_pretrained is not None:
			ts_model.load_state_dict(torch.load(best_ts_model_pretrained, map_location=device))

		# ts_model.init_params([0.01,1.8,0.02,0.9,10.0,0.2000])
		loader = DataLoader(dataset,
							batch_size=batch_size,
							shuffle=True)
		mse = nn.MSELoss(reduction='sum').to(device)

		optimizer = torch.optim.Adam(ts_model.parameters(), lr=lr)

		minloss = 1000.0

		video2pf = unet.UNetWithEmbeddingGen(in_channels=3, out_channels=3,
											 init_features=32,
											 embedding_features=embedding_features,
											 video_len=dataset.__len__() + skip_step,
											 fix_emb=False)
		video2pf.load_state_dict(torch.load(best_unet_model_pretrained))
		video2pf.eval()
		video2pf = video2pf.to(device)

		optimizer2 = torch.optim.Adam(video2pf.parameters(), lr=lr2)

		mask = torch.ones((128, 128), dtype=torch.float).to(device)
		mask[64:128, :] = 0.0

		for i in range(epoch):
				loss = 0.0
				total_size = 0
				print('epoch: ', i)
				for batch in loader:

						cv1 = batch['cv'].to(device)
						ci1 = batch['ci'].to(device)
						eta1 = batch['eta'].to(device)

						# get the movie frame
						frame1 = batch['v'].to(device)

						indicies1 = batch['index'].to(device)

						ul1 = batch['ul'].to(device)

						# get cv, ci, eta from the movie frame
						pf = video2pf(frame1, indicies1)
						cv = pf[:,0,:,:]
						ci = pf[:,1,:,:]
						eta = pf[:,2,:,:]

						# TODO: get cv, ci, eta synthesized from a random vector
						#       and add them to cv, ci, eta

						# stitch the original cv, ci, eta with the generated ones using mask
						eta = stitch_by_mask(eta, eta1, mask, ul1)

						for j in range(skip_step):
							cv_new, ci_new, eta_new = ts_model(cv, ci, eta)
							cv = cv_new
							ci = ci_new
							eta = eta_new

						cv_ref = batch['cv_ref'].to(device)
						ci_ref = batch['ci_ref'].to(device)
						eta_ref = batch['eta_ref'].to(device)

						# get the movie frame
						frame2 = batch['v_ref'].to(device)

						indicies2 = batch['index_ref'].to(device)

						ul2 = batch['ul_ref'].to(device)

						# get cv, ci, eta from the movie frame
						pf = video2pf(frame2, indicies2)
						cv_frame = pf[:,0,:,:]
						ci_frame = pf[:,1,:,:]
						eta_frame = pf[:,2,:,:]

						ul2.unsqueeze_(-1).unsqueeze_(-1)
						ul2 = ul2.repeat(1, 128, 128)
						mask2 = mask*ul2 + (1-mask)*(1-ul2)

						# TODO: get cv_frame, ci_frame, eta_frame synthesized
						#       from a random vector
						#       and add them to cv_frame, ci_frame, eta_frame

						cv_batch_loss = lambda2 * mse(cv_frame, cv_new)
						ci_batch_loss = lambda2 * mse(ci_frame, ci_new)
						eta_batch_loss = mse(mask2*eta_ref, mask2*eta_new) + \
										lambda1 * mse(mask2*eta_ref, mask2*eta_frame) + \
										lambda2 * mse(eta_frame, eta_new)

						batch_loss = cv_batch_loss + ci_batch_loss + eta_batch_loss

						optimizer.zero_grad()
						optimizer2.zero_grad()

						batch_loss.backward()

						optimizer.step()
						optimizer2.step()

						this_size = cv.size(0)
						loss += batch_loss.item()
						total_size += this_size
						ts_model.print_params(loss / total_size)
						print('(cv_ref-cv_frame)*mask=', mse(mask2*cv_ref, mask2*cv_frame).data.cpu().numpy(), \
							  '(ci_ref-ci_frame)*mask=', mse(mask2*ci_ref, mask2*ci_frame).data.cpu().numpy(), \
							  '(eta_ref-eta_frame)*mask=', mse(mask2*eta_ref, mask2*eta_frame).data.cpu().numpy())


				loss /= total_size
				print('loss: %.8f' % loss)
				if loss < minloss:
						minloss = loss
						print('Get minimal loss')
						new_record = True
						torch.save(ts_model.state_dict(), best_ts_model_out)
						torch.save(video2pf.state_dict(), best_unet_model_out)
				if i>0 and i % 100 == 0 and new_record:
					if os.path.exists(best_ts_model_out):
						shutil.copyfile(best_ts_model_out, best_ts_model_out + '-' + str(i))
						shutil.copyfile(best_unet_model_out, best_unet_model_out + '-' + str(i))
						new_record = False
				print('')

if __name__ == '__main__':
		main_v0()
