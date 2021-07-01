import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from irradiation_model import IrradiationSingleTimestep, param
import torchvision
import torchvision.io

import unet

import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(4321)
torch.cuda.manual_seed(4321)


best_ts_model_out = 'best_ts_model_out.model'
best_unet_model_out = 'best_unet_model_out.model'

data_path = '../data/irradiation_v3'
# filename_pkl = 'all_data.pkl'
video_pkl = 'synthesized_eta.pkl'

embedding_features = 8
batch_size = 64

save_data_path = '../output/irradiation_learned'
save_filename = 'frame_pred.mp4'
pkl_filename = 'frame_pred.pkl'


class IrradiationVideoDataset(Dataset):
    def __init__(self, data_path, video_pkl, skip_step=1):
        super(IrradiationVideoDataset, self).__init__()
        # self.all_data = torch.load(os.path.join(data_path, filename_pkl))
        self.video_data = torch.load(os.path.join(data_path, video_pkl))
        self.video_data = self.video_data.type(torch.float)
        self.video_data -= 128.0
        self.video_data /= 128.0
        self.start_skip = 9
        self.skip_step = skip_step
        self.cnt = len(self.video_data) -skip_step


    def __len__(self):
        return self.cnt 


def main():

    # dataset
    skip_step = 10

    dataset = IrradiationVideoDataset(data_path,
                                      # filename_pkl,
                                      video_pkl,
                                      skip_step)
    

    # load video2pf model
    video2pf = unet.UNetWithEmbeddingGen(in_channels=3, out_channels=3,
                                         init_features=32,
                                         embedding_features=embedding_features,
                                         video_len=dataset.__len__() + skip_step)
    video2pf.load_state_dict(torch.load(best_unet_model_out, map_location=device))
    video2pf.eval()
    video2pf = video2pf.to(device)

    #
    video_data = dataset.video_data[0:dataset.cnt,:,:,:].to(device)
    index_data = torch.arange(dataset.cnt).to(device)
    
    eta_pred = None
    for st in range(0, dataset.cnt, batch_size):
        print('st=', st)
                        
        # get the movie frame
        last_frame = min(st + batch_size, dataset.cnt)
        frames = video_data[st:last_frame, :, :, :]
        indicies = index_data[st:last_frame]
                        
        # get cv, ci, eta from the movie frame
        pf = video2pf(frames, indicies)
        eta = pf[:,2,:,:]

        eta = eta.detach()

        if eta_pred is None:
            eta_pred = eta
        else:
            eta_pred = torch.cat((eta_pred, eta), dim=0)

    # annotate it with red
    video_data[:,0,:,:] = video_data[:,0,:,:] + eta_pred
    video_data = torch.clamp(video_data, min=-.9999, max=.9999)

    # write it out
    video_data = video_data * 128. + 128.
    video_data = video_data.type(torch.ByteTensor)
    video_data.transpose_(1,2)
    video_data.transpose_(2,3)
    video_data = video_data.to('cpu')

    torchvision.io.write_video(os.path.join(save_data_path, save_filename), \
                               video_data, fps=24)

    video_data.transpose_(3, 2)
    video_data.transpose_(2, 1)

    torch.save(video_data, os.path.join(save_data_path, pkl_filename))
        
    

if __name__ == '__main__':
    main()
    
    
    
