import torch
import shutil
import os.path
import numpy as np
from os import path
import torch.optim as optim
from model import mipNeRF360
from config import get_config
from scheduler import lr_decay
import torch.utils.tensorboard as tb
from loss import Loss_prop,Loss_nerf,Loss_dist,mse_to_psnr
from dataset import get_dataloader, cycle


def train_model(config):
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")

    data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))
    eval_data = None
    if config.do_eval:
        eval_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))

    model = mipNeRF360(
        randomized=config.randomized,
        num_samples=config.num_samples,
        hidden_proposal=config.hidden_proposal,
        hidden_nerf=config.hidden_nerf,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        viewdir_min_deg=config.viewdir_min_deg,
        viewdir_max_deg=config.viewdir_max_deg,
        device=config.device
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    if config.continue_training:
        model.load_state_dict(torch.load(model_save_path))
        optimizer.load_state_dict(torch.load(optimizer_save_path))

    scheduler = lr_decay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    
    model.train()
    
    os.makedirs(config.log_dir, exist_ok=True)
    shutil.rmtree(path.join(config.log_dir, 'train'), ignore_errors=True)
    logger = tb.SummaryWriter(path.join(config.log_dir, 'train'), flush_secs=1)

    # 准备修改为enumerate的形式
    print_every = 100
    
    for step in range(0, config.max_steps):
        rays, pixels = next(data)
        if step % 3 == 0 or step % 3 == 1:
            t_hat,w_hat = model.prop_net.forward(rays)
            _,_,_,t,w,_ = model.nerf_net.forward(rays,t_vals=t_hat,coarse_weights=w_hat)
            t = t.detach()
            w = w.detach()

            # Compute loss and update model weights.
            loss_prop = Loss_prop(t=t,w=w,t_hat=t_hat,w_hat=w_hat)

            optimizer.zero_grad()
            loss_prop.backward()
            optimizer.step()
            scheduler.step()

        else:
            t_hat,w_hat = model.prop_net.forward(rays)
            t_hat = t_hat.detach()
            w_hat = w_hat.detach()
            final_rgbs,_,_,_,fine_weights,s_vals = model.nerf_net.forward(rays,t_vals=t_hat,coarse_weights=w_hat)
            pixels = pixels.to(config.device)

            # Compute loss and update model weights.
            loss_nerf,psnr = Loss_nerf(input=final_rgbs,target=pixels)
            loss_dist = Loss_dist(s_vals=s_vals,weights=fine_weights)
            loss_all = loss_nerf + config.dist_weight_decay * loss_dist
            
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            scheduler.step()

            psnr = psnr.detach().cpu().numpy()
            logger.add_scalar('train/loss', float(loss_all.detach().cpu().numpy()), global_step=step)
            logger.add_scalar('train/avg_psnr', float(np.mean(psnr)), global_step=step)
            logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)
            print("[step=%s]:"%(step),"coarse_psnr=%s,fine_psnr=%s,avg_psnr=%s"%( float(np.mean(psnr[:-1])),float(psnr[-1]),float(np.mean(psnr))))


            
        # if step % print_every == 0 and step != 0:
        #     print("[step=%s]:"%(step),"coarse_psnr=%s,fine_psnr=%s,avg_psnr=%s"%( float(np.mean(psnr[:-1])),float(psnr[-1]),float(np.mean(psnr))))

    #     if step % config.save_every == 0:
    #         if eval_data:
    #             del rays
    #             del pixels
    #             psnr = eval_model(config, model, eval_data)
    #             psnr = psnr.detach().cpu().numpy()
                
    #             logger.add_scalar('eval/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
    #             logger.add_scalar('eval/fine_psnr', float(psnr[-1]), global_step=step)
    #             logger.add_scalar('eval/avg_psnr', float(np.mean(psnr)), global_step=step)

    #         torch.save(model.state_dict(), path.join(config.log_dir, "model_%s.pt"%(step)))
    #         torch.save(optimizer.state_dict(), path.join(config.log_dir, "optim_%s.pt"%(step)))

    # torch.save(model.state_dict(), model_save_path)
    # torch.save(optimizer.state_dict(), optimizer_save_path)


def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _ = model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor([mse_to_psnr(torch.mean((rgb - pixels[..., :3])**2)) for rgb in comp_rgb])


if __name__ == "__main__":
    config = get_config()
    train_model(config)