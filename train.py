from e4e_projection import projection as e4e_projection
import torch
torch.backends.cudnn.benchmark = True
from util import *
from PIL import Image
import os
from torch import nn, autograd, optim
from tqdm import tqdm
from model import *
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

os.makedirs('./results', exist_ok=True)
os.makedirs('./checkpoint', exist_ok=True)

class Data(Dataset):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.transform = transforms.Compose(
            [transforms.Resize((1024,1024)),
             transforms.ToTensor()]
             # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        )
        self.device = 'cuda'
        self.image_path = os.path.join(self.path, 'real_image')
        self.images = [file for file in os.listdir(self.image_path) if file.endswith('.png')]
        if self.mode=='train':
            self.sketch_image_path = os.path.join(self.path, 'sketch_pro')
        #     self.sketch_images = [file for file in os.listdir(self.sketch_image_path) if file.endswith('.png')]

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = Image.open(os.path.join(self.image_path, image_name))

        name = strip_path_extension(image_name)+'.pt'
        if self.mode=='train':
            sketch_name = image_name[:-4] + '.PNG'
            sketch_image = Image.open(os.path.join(self.sketch_image_path, sketch_name))
            if not os.path.exists(os.path.join(self.image_path,name)):
                latent = e4e_projection(image, os.path.join(self.image_path, name), self.device)
            else:
                latent = torch.load(os.path.join(self.image_path,name))['latent']
            return latent, self.transform(image), self.transform(sketch_image)[0:3]

        else:
            latent = e4e_projection(image, os.path.join(self.image_path,name), self.device)
            return latent, self.transform(image)



class TenShot():
    def __init__(self):
        self.latent_dim=512
        self.device = 'cuda'
        self.test_dataset = Data('data/test',mode='test')
        self.testloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        self.iters = 10

        self.alpha = 0.2
        self.preserve_color = False
        self.num_iter = 500
        self.log_interval = 100

    def train(self):
        self.train_dataset = Data('data/train_chae_trace2',mode='train')
        self.trainloader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        # load Generator
        ori_generator = Generator(1024, self.latent_dim, 8, 2).to(self.device)
        ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc:storage)
        # ori_generator.load_state_dict(ckpt['g_ema'], strict=False)
        mean_latent = ori_generator.mean_latent(10000)

        generator = deepcopy(ori_generator)
        # ckpt_g = torch.load('checkpoint/ckpt_chae.pt', map_location=lambda storage, loc: storage)
        # generator.load_state_dict(ckpt_g['model'])
        ckpt_g = torch.load('checkpoint/ckpt_chae_trace.pt', map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt_g['model'])
        discriminator = Discriminator(1024,2).eval().to(self.device)
        discriminator.load_state_dict(ckpt['d'], strict=False)

        g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0,0.99))

        if self.preserve_color:
            id_swap = [9,11,15,16,17]
        else:
            id_swap = list(range(7, generator.n_latent))


        for idx in tqdm(range(self.num_iter)):
            for i, data_i in enumerate(self.trainloader):
                latent = data_i[0]
                target = data_i[2].to(self.device)
                mean_w = generator.get_latent(torch.randn([latent.size(0), self.latent_dim]).to(self.device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
                in_latent = latent.clone()
                in_latent[:,id_swap] = self.alpha*latent[:,id_swap] + (1-self.alpha)*mean_w[:,id_swap]

                img = generator(in_latent, input_is_latent=True)

                with torch.no_grad():
                    real_feat = discriminator(target)
                fake_feat = discriminator(img)

                loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)

                if idx%50==0:
                    print(f"Idx: {idx}, Loss: {loss}")
                    output_n = f'./results/train_img/{idx}_{i}_output.png'
                    gt_n = f'./results/train_img/{idx}_{i}_gt.png'
                    torchvision.utils.save_image(img, output_n)
                    torchvision.utils.save_image(target, gt_n)

                g_optim.zero_grad()
                loss.backward()
                g_optim.step()

        torch.save({
            'model': generator.state_dict(),
            'optimizer': g_optim.state_dict()
        }, 'checkpoint/ckpt_chae_trace2.pt')



    def eval(self):
        generator = Generator(1024, self.latent_dim, 8, 2).to(self.device)
        ckpt = torch.load('checkpoint/ckpt_chae_trace2.pt', map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt['model'])
        os.makedirs('./results/ckpt_chae_trace2', exist_ok=True)
        os.makedirs('./results/ckpt_chae_trace2/sketch', exist_ok=True)
        os.makedirs('./results/ckpt_chae_trace2/img', exist_ok=True)
        for i, data_i in enumerate(self.testloader):
            latent = data_i[0]
            img = data_i[1].to(self.device)
            with torch.no_grad():
                output = generator(latent, input_is_latent=True)
            sketch_name = f"./results/ckpt_chae_trace2/sketch/{i}.png"
            img_name = f"./results/ckpt_chae_trace2/img/{i}.png"
            total_name = f"./results/ckpt_chae_trace2/{i}.png"
            torchvision.utils.save_image(torchvision.utils.make_grid([img.squeeze(0), output.squeeze(0)], nrow=2), total_name)
            torchvision.utils.save_image(output, sketch_name)
            torchvision.utils.save_image(torchvision.utils.make_grid(img.squeeze(0)), img_name)

if __name__=='__main__':
    tenshot = TenShot()
    # tenshot.train()
    tenshot.eval()

