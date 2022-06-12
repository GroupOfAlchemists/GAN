import argparse
import os
import random
import numpy
import torch
import logging
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import copy
from Networks import Generator, Discriminator
from Loss import ZeroCenteredGradientPenalty, RelativisticLoss

ema_beta = 0.998 # ffhq: 0.99778438712388889017237329703832 cifar: 0.99991128109664301904760707704894
gamma = 0.5
grad_accum = 8

img_nrow = 8
manualSeed = 42

def run():
    torch.set_printoptions(threshold=1)
    logging.basicConfig(level=logging.INFO, filename='train_log.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    
    model = None #torch.load('epoch_36.pth', map_location='cpu')
    
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    numpy.random.seed(manualSeed)

    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset = dset.ImageFolder('./Data',transform=transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5)
    ]))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize * 2,
                                             shuffle=True, num_workers=int(opt.workers), drop_last=True)
    
    device = torch.device("cuda:0")
    nz = int(opt.nz)
    
    fixed_noise = torch.randn(opt.batchSize, nz, device=device)
    
    netG = Generator(NoiseDimension=nz, StageWidths=[1024, 512, 512, 256, 128], BlocksPerStage=[3, 3, 3, 3, 3]).to(device)
    netD = Discriminator(StageWidths=[128, 256, 512, 512, 1024], BlocksPerStage=[3, 3, 3, 3, 3]).to(device)
    G_ema = copy.deepcopy(netG).eval()
    
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0, 0.99))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0, 0.99))
    
    if model is not None:
        fixed_noise = model['fixed_noise'].to(device)
        netG.load_state_dict(model['g_state_dict'], strict=True)
        netD.load_state_dict(model['d_state_dict'], strict=True)
        G_ema.load_state_dict(model['g_ema_state_dict'], strict=True)
        optimizerD.load_state_dict(model['optimizerD_state_dict'])
        optimizerG.load_state_dict(model['optimizerG_state_dict'])
     
    print('G params: ' + str(sum(p.numel() for p in netG.parameters() if p.requires_grad)))
    print('D params: ' + str(sum(p.numel() for p in netD.parameters() if p.requires_grad)))
    
    chunk_size = opt.batchSize // grad_accum
    
    def D_backward(real):
        real.requires_grad = True
        
        noise = torch.randn(chunk_size, nz, device=device)
        fake = netG(noise)
        
        output_r = netD(real)
        output_f = netD(fake)
        
        r1_penalty = ZeroCenteredGradientPenalty(real, output_r)
        r2_penalty = ZeroCenteredGradientPenalty(fake, output_f)
        
        errD = RelativisticLoss(output_r, output_f) + gamma * (r1_penalty + r2_penalty)
        (errD / grad_accum).backward()
        
        return [x.detach().item() / grad_accum for x in [errD, r1_penalty, r2_penalty]]
    
    def G_backward(real):
        noise = torch.randn(chunk_size, nz, device=device)
        fake = netG(noise)
        
        output_f = netD(fake)
        output_r = netD(real)
        
        errG = RelativisticLoss(output_f, output_r)
        (errG / grad_accum).backward()
        
        return errG.detach().item() / grad_accum
    
    for epoch in range(0 if model is None else model['epoch'] + 1, 1000000):
        for i, data in enumerate(dataloader, 0):
            netD.requires_grad = True
            netG.requires_grad = False
            
            netD.zero_grad()
            
            real = data[0][0 : opt.batchSize, :, :, :].to(device)
            
            errD = 0
            r1_penalty = 0
            r2_penalty = 0
            
            for x in range(grad_accum):
                errD_chunk, r1_penalty_chunk, r2_penalty_chunk = D_backward(real[x * chunk_size : (x + 1) * chunk_size, :, :, :])
                errD += errD_chunk
                r1_penalty += r1_penalty_chunk
                r2_penalty += r2_penalty_chunk
            
            optimizerD.step()

            ###########################
            
            netD.requires_grad = False
            netG.requires_grad = True

            netG.zero_grad()
            
            real = data[0][opt.batchSize : 2 * opt.batchSize, :, :, :].to(device)
            
            errG = 0
            
            for x in range(grad_accum):
                errG_chunk = G_backward(real[x * chunk_size : (x + 1) * chunk_size, :, :, :])
                errG += errG_chunk
                
            optimizerG.step()
            
            ###########################
            
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), netG.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), netG.buffers()):
                    b_ema.copy_(b)
                    
            ###########################
            
            log_str = '[%d][%d/%d] Loss_D: %.4f Loss_G: %.4f R1: %.4f R2: %.4f' % (epoch, i, len(dataloader), errD, errG, gamma * r1_penalty, gamma * r2_penalty)
            print(log_str)
            logging.info(log_str)
            
            ###########################
            
            if i % 100 == 0:
                fake = []
                with torch.no_grad():
                    for x in range(grad_accum):
                        fake += [G_ema(fixed_noise[x * chunk_size : (x + 1) * chunk_size, :])]
                    noise = torch.randn(10000, nz, device=device)
                    w = G_ema.LatentLayer(noise).mean(0).view(1, -1)
                    mean = G_ema(w, EnableLatentMapping=False)
                fake = torch.cat(fake, 0)
                vutils.save_image(real, '%s/real_samples.png' % opt.outf, normalize=True, nrow=img_nrow)
                vutils.save_image(torch.clamp(fake, -1, 1), '%s/fake_samples_epoch_%04d_%04d.png' % (opt.outf, epoch, i), normalize=True, nrow=img_nrow)
                vutils.save_image(torch.clamp(mean, -1, 1), '%s/mean_sample_epoch_%04d_%04d.png' % (opt.outf, epoch, i), normalize=True, nrow=1)

        torch.save({
            'epoch': epoch,
            'g_ema_state_dict': G_ema.state_dict(),
            'g_state_dict': netG.state_dict(),
            'd_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'fixed_noise': fixed_noise,
            'loss_D': errD,
            'loss_G': errG,
            'r1_penalty': gamma * r1_penalty,
            'r2_penalty': gamma * r2_penalty,
            }, '%s/epoch_%d.pth' % (opt.outf, epoch))
        
if __name__ == '__main__':
    run()