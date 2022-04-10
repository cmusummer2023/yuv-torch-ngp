import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from tensoRF.utils import *

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    ### training options
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--l1_reg_weight', type=float, default=4e-5)
    # (only valid when not using --cuda_ray)
    parser.add_argument('--num_steps', type=int, default=512)
    parser.add_argument('--upsample_steps', type=int, default=0)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--cp', action='store_true', help="use TensorCP instead of TensorVMSplit")
    parser.add_argument('--resolution0', type=int, default=128)
    parser.add_argument('--resolution1', type=int, default=300)
    parser.add_argument("--upsample_model_steps", type=int, action="append", default=[2000, 3000, 4000, 5500, 7000])
    ### dataset options
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (default is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable.")
    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1024, help="GUI width")
    parser.add_argument('--H', type=int, default=1024, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()
    print(opt)
    
    seed_everything(opt.seed)

    if opt.cp:
        from tensoRF.network_cp import NeRFNetwork
    else:
        from tensoRF.network import NeRFNetwork

    model = NeRFNetwork(
        resolution=[opt.resolution0] * 3,
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1 if opt.mode == 'colmap' else 100,
    )
    
    print(model)

    criterion = torch.nn.MSELoss()

    ### test mode
    if opt.test:

        trainer = Trainer('ngp', opt, model, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter()], use_checkpoint='latest')

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=not opt.preload)

            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.

            trainer.save_mesh(resolution=256, threshold=0.1)
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(2e-2, 1e-3), betas=(0.9, 0.99), eps=1e-15)

        # need different milestones for GUI/CMD mode.
        scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000] if opt.gui else [100, 200], gamma=0.33)

        trainer = Trainer('ngp', opt, model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, metrics=[PSNRMeter()], use_checkpoint='latest', eval_interval=50)

        # calc upsample target resolutions
        upsample_resolutions = (np.round(np.exp(np.linspace(np.log(opt.resolution0), np.log(opt.resolution1), len(opt.upsample_model_steps) + 1)))).astype(np.int32).tolist()[1:]
        print('upsample_resolutions:', upsample_resolutions)
        trainer.upsample_resolutions = upsample_resolutions

        if opt.gui:
            train_dataset = NeRFDataset(opt.path, type='all', mode=opt.mode, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=not opt.preload)
            trainer.train_loader = train_loader # attach dataloader to trainer

            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=not opt.preload)
            valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode, downscale=2, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, pin_memory=not opt.preload)

            trainer.train(train_loader, valid_loader, 300)

            # also test
            test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=not opt.preload)
            
            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.

            trainer.save_mesh(resolution=256, threshold=0.1)