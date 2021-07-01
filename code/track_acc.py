import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from irradiation_model import IrradiationSingleTimestep, param
import torchvision
import torchvision.io
import csv

import unet

import math
import os
import sys

if len(sys.argv) > 1:
	run_id = sys.argv[1]
else:
	run_id = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(4321)
torch.cuda.manual_seed(4321)


best_unet_model_out = 'best_unet_model_out.model'
baseline_unet_model = 'best_video_train_only_unet_ef8.model'

data_path = '../data/irradiation_v4'
filename_pkl = 'all_data_test.pkl'
video_pkl = 'synthesized_eta_test.pkl'

embedding_features = 8
batch_size = 64
bound = 0.5

save_data_path = '../data/irradiation_v4'


class IrradiationVideoDataset(Dataset):
	def __init__(self, data_path, filename_pkl, video_pkl, skip_step=1):
		super(IrradiationVideoDataset, self).__init__()
		self.all_data = torch.load(os.path.join(data_path, filename_pkl))
		self.video_data = torch.load(os.path.join(data_path, video_pkl))
		self.video_data = self.video_data.type(torch.float)
		self.video_data -= 128.0
		self.video_data /= 128.0
		self.start_skip = 9
		self.cnt = len(self.video_data)


	def __len__(self):
		return self.cnt


def main():

	# dataset
	skip_step = 10
	total_frame = 3487

	dataset = IrradiationVideoDataset(data_path,
									  filename_pkl,
									  video_pkl,
									  skip_step)

	print("data_path,filename_pkl,video_pkl",data_path,filename_pkl,video_pkl)
	# load video2pf model
	video2pf = unet.UNetWithEmbeddingGen(in_channels=3, out_channels=3,
                                        init_features=32,
                                        embedding_features=embedding_features,
                                        video_len=total_frame,
                                        fix_emb=False)
	video2pf.load_state_dict(torch.load(best_unet_model_out, map_location=device))
	video2pf.eval()
	video2pf = video2pf.to(device)

	video2pf_base = unet.UNetWithEmbeddingGen(in_channels=3, out_channels=3,
                                        init_features=32,
                                        embedding_features=embedding_features,
                                        video_len=total_frame,
                                        fix_emb=False)
	video2pf_base.load_state_dict(torch.load(baseline_unet_model, map_location=device))
	video2pf_base.eval()
	video2pf_base = video2pf_base.to(device)

	#
	video_data = dataset.video_data[0:dataset.cnt,:,:,:].to(device)
	index_data = torch.arange(dataset.cnt).to(device)

	# extract the valid eta values, 
	# ignoring the dummy frames at the beginning and end of all_data
	all_data_valid = dataset.all_data[dataset.start_skip:dataset.start_skip+dataset.cnt]
	all_eta = torch.cat([x['eta'][1:-1,1:-1].unsqueeze(0) for x in all_data_valid])
	eta_pred = None
	all_acc = 0.0
	all_base_acc = 0.0
	cnt = 0.0

	print("all_eta=",all_eta.size())
	for st in range(0, dataset.cnt, batch_size):
		print('st=', st)

		# get the movie frame
		last_frame = min(st + batch_size, dataset.cnt)
		frames = video_data[st:last_frame, :, :, :]
		indicies = index_data[st:last_frame]

		# get cv, ci, eta from the movie frame
		pf = video2pf(frames, indicies)
		eta = pf[:,2,:,:]
		eta = eta.detach().cpu()
		eta = torch.ge(eta, bound)

		pf_base = video2pf_base(frames, indicies)
		eta_base = pf_base[:, 2, :, :]
		eta_base= eta_base.detach().cpu()
		eta_base = torch.ge(eta_base, bound)

		eta_ref = all_eta[st:last_frame, :, :]
		eta_ref = torch.ge(eta_ref, bound)

		img_size = eta.size(1)*eta.size(2)

		for idx in range(eta.size(0)):
			acc = torch.eq(eta[idx], eta_ref[idx]).sum().item() / img_size
			base_acc = torch.eq(eta_base[idx], eta_ref[idx]).sum().item() / img_size
			# csv_writer.writerow([str(acc), str(base_acc)])
			cnt += 1.0
			all_acc += acc
			all_base_acc += base_acc



	print('Ours: %.8f' % (all_acc / cnt))
	print('Baseline: %.8f' % (all_base_acc / cnt))


if __name__ == '__main__':
	main()



