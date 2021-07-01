import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from irradiation_model import IrradiationSingleTimestep, param
import torchvision

import unet

import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(4321)
torch.cuda.manual_seed(4321)



best_ts_model_out = 'best_ts_model_out.model'
best_unet_model_out = 'best_unet_model_out.model'

data_path = '../data/irradiation_v3'
filename_pkl = 'all_data.pkl'
video_pkl = 'synthesized_eta.pkl'

embedding_features = 8

save_data_path = '../data/irradiation'
eta_filename = 'dynamics_eta.mp4'
eta_pkl = 'dynamics_eta.pkl'


def write_video(all_data, which, filename, pkl_filename):
    all_phi = [all_data[i][which] for i in range(len(all_data))]
    # print('all_phi=', all_phi)

    all_phi_T = torch.stack(all_phi, dim=0)

    # all_phi_T = all_phi_T[:10000,:,:]

    print('max(all_phi_T)=', torch.max(all_phi_T))
    print('min(all_phi_T)=', torch.min(all_phi_T))

    print('all_phi_T.size()=', all_phi_T.size())

    all_phi_T = all_phi_T.to(device)

    in_eta = (torch.randn(all_phi_T.size()) * 0.2528152 + 0.05601295).to(device)
    out_eta = (torch.randn(all_phi_T.size()) * 0.21414237 - 0.20813818).to(device)

    v = all_phi_T * in_eta + (1 - all_phi_T) * out_eta

    avg_weight = torch.ones(19, 3, 3) / 19. / 3 / 3.
    avg_weight = avg_weight.to(device)
    avg_weight.unsqueeze_(dim=0)
    avg_weight.unsqueeze_(dim=0)
    v.unsqueeze_(dim=0)
    v.unsqueeze_(dim=0)
    v = torch.nn.functional.conv3d(v, avg_weight)
    v.squeeze_(dim=0)
    v.squeeze_(dim=0)


    v = (v * 128.) + 128.

    v = torch.unsqueeze(v, dim=-1)
    print('v.size=', v.size())

    v = v.to(torch.int).type(torch.ByteTensor)

    v_C = v.repeat([1, 1, 1, 3])
    print('v_C.size=', v_C.size())

    v_C = v_C.to('cpu')

    torchvision.io.write_video(filename, \
                               v_C, fps=24)

    v_C.transpose_(3, 2)
    v_C.transpose_(2, 1)

    torch.save(v_C, pkl_filename)


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
                'index_ref':index+self.skip_step
        }

    def __len__(self):
        return self.cnt 


def main():
    # load ts_model
    ts_model = IrradiationSingleTimestep()
    ts_model.load_state_dict(torch.load(best_ts_model_out, map_location=device))
    ts_model.eval()
    ts_model = ts_model.to(device)
    print('ts_model params:')
    ts_model.print_params()

    # dataset
    skip_step = 10

    dataset = IrradiationVideoDataset(data_path,
                                      filename_pkl,
                                      video_pkl,
                                      skip_step)
    loader = DataLoader(dataset,
                        batch_size=param.batch_size,
                        shuffle=True)
    

    # load video2pf model
    video2pf = unet.UNetWithEmbeddingGen(in_channels=3, out_channels=3,
                                         init_features=32,
                                         embedding_features=embedding_features,
                                         video_len=dataset.__len__() + skip_step)
    video2pf.load_state_dict(torch.load(best_unet_model_out, map_location=device))
    video2pf.eval()
    video2pf = video2pf.to(device)

    # first frame
    item = dataset.__getitem__(0)
    frame0 = item['v'].unsqueeze_(0).to(device)
    index0 = torch.tensor(item['index']).unsqueeze_(0).to(device)

    pf0 = video2pf(frame0, index0)
    cv = pf0[:,0,:,:].squeeze_(0)
    ci = pf0[:,1,:,:].squeeze_(0)
    eta = pf0[:,2,:,:].squeeze_(0)

    # simulate the dynamics
    all_data = []
    ttime = 0
    for step in range(param.nstep):
        cv_new, ci_new, eta_new = ts_model(cv, ci, eta)
        del cv
        del ci
        del eta
        cv, ci, eta = cv_new, ci_new, eta_new
        cv = cv.detach()
        ci = ci.detach()
        eta = eta.detach()
        all_data.append({'step': step, 'cv': cv, 'ci': ci, 'eta': eta})

        if step % param.nprint == 0:
            print('Step %d, time %.2f' % (step, ttime))

        ttime += param.dt

    # torch.save(all_data, os.path.join(save_data_path, save_filename_pkl))
    write_video(all_data, 'eta', os.path.join(save_data_path, eta_filename),
                os.path.join(save_data_path, eta_pkl))

    

if __name__ == '__main__':
    main()
    
    
    
