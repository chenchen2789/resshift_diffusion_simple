import os
from copy import deepcopy
import numpy as np
import torch
import tqdm
import math
import cv2
import torch.nn.functional as F
import torchvision as thv
from Diffusion.Diffusion_ResShift import DiffusionResShift
from ldm.models.autoencoder import VQModelTorch
from arch.unet import UNetModelSwin
from dataset_loader.dataloader import DataloaderSimpleTest
from torch.utils.data import DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt



class ResShiftTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = self.configs.train['epochs']
        self.num_timesteps = self.configs.diffusion.params.get("steps")
        self.diffusion_sf = self.configs.diffusion.params.get("sf")
        self.diffusion_scale_factor = self.configs.diffusion.params.get("scale_factor")

        self.train_dataloader = self.build_training_dataloader()
        self.val_dataloader = self.build_val_dataloader()
        self.build_model()
        self.build_diffusion_model()
        self.setup_optimization()



    def setup_optimization(self):
        self.optimizer = torch.optim.Adam(self.unet_model.parameters(), lr=self.configs.train.get('lr'))


    def build_model(self):
        params = self.configs.model.get('params', dict)
        unet_model = UNetModelSwin(**params)
        unet_model.cuda()
        if self.configs.model.ckpt_path is not None:
            state = torch.load(self.configs.model.ckpt_path, map_location=f"cuda:0")
            if 'state_dict' in state:
                state = state['state_dict']
            reload_model(unet_model, state)
        self.unet_model = unet_model

        if self.configs.autoencoder is not None:
            print('using autoencoder')
            params_autoencoder = self.configs.autoencoder.get('params', dict)
            autoencoder = VQModelTorch(**params_autoencoder)
            autoencoder.cuda()

            state = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:0")
            if 'state_dict' in state:
                state = state['state_dict']
            reload_model(autoencoder, state)

            self.freeze_model(autoencoder)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None


    def build_diffusion_model(self):
        diffusion_opt = self.configs.get('diffusion', dict)
        self.DiffusionResShift_Model = DiffusionResShift(diffusion_opt)



    def build_training_dataloader(self):
        opt = {}
        opt['paths'] = self.configs.data.train.params['dir_paths']
        opt['sf'] = self.configs.diffusion.params.get("sf")
        opt['gt_size'] = self.configs.data.train.params.get('gt_size')
        batch_size = self.configs.train.get('batch')[0]
        num_workers = self.configs.train.get('num_workers')
        return DataLoader(DataloaderSimpleTest(opt), batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def build_val_dataloader(self):
        opt = {}
        opt['paths'] = self.configs.data.val.params['dir_paths']
        opt['sf'] = self.configs.diffusion.params.get("sf")
        opt['gt_size'] = self.configs.data.train.params.get('gt_size')
        batch_size = self.configs.train.get('batch')[1]
        num_workers = self.configs.train.get('num_workers')
        return DataLoader(DataloaderSimpleTest(opt), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def train_step(self, epoch):
        batch = -1
        for data in self.train_dataloader:
            batch += 1
            current_batchsize = data['gt'].shape[0]
            micro_batchsize = self.configs.train.microbatch
            num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)
            self.optimizer.zero_grad()
            loss_vis = [0 for _ in range(self.num_timesteps)]
            num_loss_vis = [1 for _ in range(self.num_timesteps)]
            for jj in range(0, current_batchsize, micro_batchsize):
                ##数据准备##
                micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
                gt, lq = micro_data['gt'].to(self.device), micro_data['lq'].to(self.device)
                lq_ori = deepcopy(lq)

                if self.autoencoder is not None:
                    with torch.no_grad():
                        if self.diffusion_sf != 1:
                            lq = F.interpolate(lq, scale_factor=self.diffusion_sf, mode='bicubic')
                            gt, lq = self.autoencoder.encode(gt)*self.diffusion_scale_factor, self.autoencoder.encode(lq)*self.diffusion_scale_factor
                        else:
                            gt, lq = self.autoencoder.encode(gt), self.autoencoder.encode(lq)

                tt = torch.randint(
                    0, self.num_timesteps,
                    size=(lq.shape[0],),
                    device=lq.device,
                )
                noise = torch.randn(
                    size= lq.shape,
                    device=lq.device,
                )

                ##网络模型预测##
                x_t = self.DiffusionResShift_Model.forward_addnoise(x_start=gt, y=lq, t=tt, noise=noise)
                network_output = self.unet_model(self.DiffusionResShift_Model.scale_input_resshift(inputs=x_t, t=tt), timesteps=tt, lq=lq_ori)

                ##计算Losss##
                error = (gt - network_output)**2
                loss = error.mean(dim=list(range(1, len(error.shape))))

                for i in range(len(loss)):
                    loss_vis[tt[i]] += loss[i].item()
                    num_loss_vis[tt[i]] += 1
                loss_vis = [loss_vis[i]/num_loss_vis[i] for i in range(len(loss_vis))]
                loss = torch.mean(loss)/num_grad_accumulate

                loss.backward()

            self.optimizer.step()

            if batch % self.configs.train.log_freq[0] == 0:
                print(f"Epoch {epoch}, Batch {batch}, Loss {loss.item()}, loss_step:{loss_vis}")
                with open('./log/loss.txt', 'a') as f:
                    f.write(f"Epoch {epoch}, Batch {batch}, Loss {loss.item()}, loss_step:{loss_vis}\n")



    def train(self):
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(self.epochs, epoch))
            self.train_step(epoch)
            if epoch % self.configs.train.log_freq[1] == 0:
                print(f"evaluating epoch {epoch}")
                self.evaluate()
            if epoch % self.configs.train.log_freq[2] == 0:
                torch.save(self.unet_model.state_dict(), f'./log/{epoch}.pth')

    def evaluate(self):
        for data in self.val_dataloader:
            lq = data['lq'].to(self.device)
            lq2 = deepcopy(lq)

            ##encoder编码
            if self.autoencoder is not None:
                with torch.no_grad():
                    if self.diffusion_sf != 1:
                        lq = F.interpolate(lq, scale_factor=self.diffusion_sf, mode='bicubic')
                        lq = self.autoencoder.encode(lq) * self.diffusion_scale_factor
                    else:
                        lq = self.autoencoder.encode(lq)
            else:
                print('without autoencoder')


            indices = list(range(self.num_timesteps))[::-1]
            noise = torch.randn_like(lq)
            x_t = self.DiffusionResShift_Model.prior_sample(lq, noise)

            indices = tqdm.auto.tqdm(indices)
            results = []
            for t in indices:
                tt = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                with torch.no_grad():
                    x_pred = self.unet_model(self.DiffusionResShift_Model.scale_input_resshift(inputs=x_t, t=tt), timesteps=tt, lq=lq2)
                    noise = torch.randn_like(x_pred)
                    x_t = self.DiffusionResShift_Model.inverse_denoise(x_start=x_pred, x_t=x_t, t=tt, noise=noise)
                    results.append(deepcopy(x_t))

            for i in range(len(results)):
                img = results[i]
                if self.autoencoder is not None:
                    with torch.no_grad():
                        if self.diffusion_sf != 1:
                            img = self.autoencoder.decode((1.0/self.diffusion_scale_factor)*img)
                        else:
                            img = self.autoencoder.decode(img)
                results[i] = img.clamp(min=-1.0, max=1.0)
            results.append(lq2)
            plt.figure(1, dpi=500)
            for i in range(len(results)):
                for j in range(len(results[0])):
                    img = results[i][j] * 0.5 + 0.5
                    img = img.cpu().numpy()
                    img = np.transpose(img, (1,2,0))
                    plt.subplot(len(results[0]), len(results), i + 1 + j * len(results))
                    plt.imshow(img)
                    plt.xticks([]), plt.yticks([])
                    if j == 0:
                        if i != 4:
                            plt.title(f"step: {i}")
                        else:
                            plt.title(f"lq")
            plt.suptitle('reconstruction of xt in different step')
            plt.tight_layout()
            plt.show()
            return



    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False


def reload_model(model, ckpt):
    module_flag = list(ckpt.keys())[0].startswith('module.')
    compile_flag = '_orig_mod' in list(ckpt.keys())[0]

    for source_key, source_value in model.state_dict().items():
        target_key = source_key
        if compile_flag and (not '_orig_mod.' in source_key):
            target_key = '_orig_mod.' + target_key
        if module_flag and (not source_key.startswith('module')):
            target_key = 'module.' + target_key

        assert target_key in ckpt
        source_value.copy_(ckpt[target_key])



