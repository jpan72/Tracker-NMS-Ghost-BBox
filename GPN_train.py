import argparse
import json
import time

import test  
from models import *
from utils.datasets import GhostDataset, collate_fn
from utils.utils import *
from utils.log import logger
from torchvision.transforms import transforms as T
from tensorboardX import SummaryWriter
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(
        cfg,
        data_cfg,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None,
):
    weights = 'weights' 
    mkdir_if_missing(weights)
    latest = osp.join(weights, 'latest.pt')

    torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    cfg_dict = parse_model_cfg(cfg) 
    img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # Get dataloader

    if opt.network == 'alexnet':
        input_size = 256
    else:
        input_size = 224
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.RandomCrop(200),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # dataset = JointDataset(dataset_root, trainset_paths, img_size, augment=True, transforms=transforms)

    # dataset_root = '../preprocess-ghost-bbox-HA0.3/MOT17/MOT17/train'
    # dataset_root = '../preprocess-ghost-bbox-th0.6/MOT17/MOT17/train'
    dataset_root = '../preprocess-ghost-bbox-th0.6-map-more-filter/MOT17/MOT17/train'
    dataset = GhostDataset(dataset_root, transforms)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
    #                                          num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)


    # dataset_root_test = '../preprocess-ghost-bbox-th0.6/2DMOT2015/train'
    dataset_root_test = '../preprocess-ghost-bbox-th0.6-map-more-filter/2DMOT2015/train'
    dataset_test = GhostDataset(dataset_root_test, transforms)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)

    # Initialize model
    # model = Darknet(cfg_dict, dataset.nID)
    gpn = GPN(network=opt.network).cuda()

    if opt.resume:
        gpn.load_state_dict(torch.load(opt.load_path))

    # from torch.nn import init
    # for layer_p in gpn._all_weights:
    #     for p in layer_p:
    #         if 'weight' in p:
    #             # print(p, a.__getattr__(p))
    #             init.normal(gpn.__getattr__(p), 0.0, 0.02)
    #             # print(p, a.__getattr__(p))

    else:
        import torch.nn.init as weight_init
        for name, param in gpn.named_parameters():
            if 'weight' in name:
                weight_init.normal(param);


    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0

    # if resume:
    #     checkpoint = torch.load(latest, map_location='cpu')
    #
    #     # Load weights to resume from
    #     model.load_state_dict(checkpoint['model'])
    #     model.cuda().train()
    #
    #     # Set optimizer
    #     optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9)
    #
    #     start_epoch = checkpoint['epoch'] + 1
    #     if checkpoint['optimizer'] is not None:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #
    #     del checkpoint  # current, saved
    #
    # else:
    #     # Initialize model with backbone (optional)
    #     if cfg.endswith('yolov3.cfg'):
    #         load_darknet_weights(model, osp.join(weights ,'darknet53.conv.74'))
    #         cutoff = 75
    #     elif cfg.endswith('yolov3-tiny.cfg'):
    #         load_darknet_weights(model, osp.join(weights , 'yolov3-tiny.conv.15'))
    #         cutoff = 15
    #
    #     model.cuda().train()
    #
    #     # Set optimizer
    #     optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9, weight_decay=1e-4)

    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, gpn.parameters()), lr=opt.lr, momentum=.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, gpn.parameters()), lr=opt.lr)
    smooth_l1_loss = nn.SmoothL1Loss().cuda()
    smooth_l1_loss_test = nn.SmoothL1Loss(reduction='sum').cuda()
    # model = torch.nn.DataParallel(model)

    # # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #         milestones=[int(0.5*opt.epochs), int(0.75*opt.epochs)], gamma=0.1)
    
    # # An important trick for detection: freeze bn during fine-tuning
    # if not opt.unfreeze_bn:
    #     for i, (name, p) in enumerate(model.named_parameters()):
    #         p.requires_grad = False if 'batch_norm' in name else True

    # model_info(model)

    model_info(gpn)

    f = open("weights.txt", "w")
    f.close()
    f = open("weights.txt", "a")
    exp_name = f'{datetime.now():%Y-%m-%d-%H:%M%z}'
    writer = SummaryWriter(osp.join('../exp-ghost-bbox', exp_name))
    t0 = time.time()
    for epoch in range(epochs):
        print('Epoch {}'.format(epoch))

        # training
        gpn.train()
        epoch += start_epoch

        # logger.info(('%8s%12s' + '%10s' * 6) % (
        #     'Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time'))
        
        # # Freeze darknet53.conv.74 for first epoch
        # if freeze_backbone and (epoch < 2):
        #     for i, (name, p) in enumerate(model.named_parameters()):
        #         if int(name.split('.')[2]) < cutoff:  # if layer < 75
        #             p.requires_grad = False if (epoch == 0) else True

        ui = -1
        # rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        # for i, (imgs, targets, _, _, targets_len) in enumerate(dataloader):
        print(len(dataloader))
        for i, (track_imgs, det_imgs, track_tlbrs, det_tlbrs, tlwh_histories, target_delta_bbox) in enumerate(dataloader):
        # for i, (track_imgs, track_tlbrs, tlwh_histories, target_delta_bbox) in enumerate(dataloader):
            n_iter = epoch * len(dataloader) + i

            # if i % 100 == 0:
            #     for layer, param in enumerate(gpn.parameters()):
            #         if layer in [22, 24]:
            #             # print(param.data.size())
            #
            #             if layer == 22:
            #                 print(param.data[2,:])
            #                 # pass
            #             else:
            #                 # print(param.data[:8])
            #                 pass
            #             # print()

            track_imgs = track_imgs.cuda().float()
            det_imgs = det_imgs.cuda().float()
            track_tlbrs = track_tlbrs.cuda().float()
            det_tlbrs = det_tlbrs.cuda().float()
            tlwh_histories = tlwh_histories.cuda().float()
            tlwh_histories[:,:,0] /= 1088
            tlwh_histories[:,:,1] /= 608
            tlwh_histories[:,:,2] /= 1088
            tlwh_histories[:,:,3] /= 608
            target_delta_bbox = target_delta_bbox.cuda().float()

            # # SGD burn-in
            # burnin = min(1000, len(dataloader))
            # if (epoch == 0) & (i <= burnin):
            #     lr = opt.lr * (i / burnin) **4
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr
            
            # Compute loss, compute gradient, update parameters
            # loss, components = model(imgs.cuda(), targets.cuda(), targets_len.cuda())
            # components = torch.mean(components.view(-1, 5),dim=0)

            delta_bbox = gpn(track_imgs, det_imgs, track_tlbrs, det_tlbrs, tlwh_histories)
            # delta_bbox = gpn(track_imgs, track_tlbrs,  tlwh_histories)
            loss = smooth_l1_loss(delta_bbox, target_delta_bbox)

            # loss = torch.mean(loss)
            loss.backward()
            # print(loss)
            writer.add_scalar('train/loss', loss.cpu().detach().numpy(), n_iter)

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            
            # for ii, key in enumerate(model.module.loss_names):
            #     rloss[key] = (rloss[key] * ui + components[ii]) / (ui + 1)

            # s = ('%8s%12s' + '%10.3g' * 6) % (
            #     '%g/%g' % (epoch, epochs - 1),
            #     '%g/%g' % (i, len(dataloader) - 1),
            #     rloss['box'], rloss['conf'],
            #     rloss['id'],rloss['loss'],
            #     rloss['nT'], time.time() - t0)
            t0 = time.time()
            # if i % opt.print_interval == 0:
            #     logger.info(s)


            if i %100 == 0:
                target_delta_bbox[:,0] *= 1088
                target_delta_bbox[:,1] *= 608
                target_delta_bbox[:,2] *= 1088
                target_delta_bbox[:,3] *= 608

                delta_bbox[:,0] *= 1088
                delta_bbox[:,1] *= 608
                delta_bbox[:,2] *= 1088
                delta_bbox[:,3] *= 608

                print()
                print('delta bbox and target')
                print(delta_bbox)
                print(target_delta_bbox)
                print(loss)
                # import pdb; pdb.set_trace()

        # # Save latest checkpoint
        # checkpoint = {'epoch': epoch,
        #               # 'model': gpn.module.state_dict(),
        #               'model': gpn.state_dict(),
        #               'optimizer': optimizer.state_dict()}
        # torch.save(checkpoint, latest)


        # # Calculate mAP
        # if epoch % opt.test_interval ==0:
        #     with torch.no_grad():
        #         mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)
        #         test.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)


        # # Call scheduler.step() after opimizer.step() with pytorch > 1.1.0
        # scheduler.step()


        # test

        # gpn.eval()
        #
        # loss_test_sum = 0
        # loss_num_data = 0
        #
        # for i, (track_feat, det_feat, target_delta_bbox) in enumerate(dataloader_test):
        #     loss_num_data += track_feat.size(0)
        #
        #     track_feat = track_feat.cuda().float()
        #     det_feat = det_feat.cuda().float()
        #     target_delta_bbox = target_delta_bbox.cuda().float()
        #
        #     delta_bbox = gpn(track_feat, det_feat)
        #     loss = smooth_l1_loss_test(delta_bbox, target_delta_bbox)
        #     loss_test_sum += loss.cpu().detach().numpy()
        #
        # loss_test_mean = loss_test_sum / loss_num_data
        # writer.add_scalar('test/loss', loss_test_mean, n_iter)


        gpn.eval()

        loss_test_sum = 0

        print()
        print('========= test delta boxes and target delta boxes:==========')
        for i, (track_imgs, det_imgs, track_tlbrs, det_tlbrs, tlwh_histories, target_delta_bbox) in enumerate(dataloader_test):
        # for i, (track_imgs, track_tlbrs,  tlwh_histories, target_delta_bbox) in enumerate(dataloader_test):

            track_imgs = track_imgs.cuda().float()
            det_imgs = det_imgs.cuda().float()
            track_tlbrs = track_tlbrs.cuda().float()
            det_tlbrs = det_tlbrs.cuda().float()
            tlwh_histories = tlwh_histories.cuda().float()
            target_delta_bbox = target_delta_bbox.cuda().float()

            delta_bbox = gpn(track_imgs, det_imgs, track_tlbrs, det_tlbrs, tlwh_histories)
            # print()
            loss = smooth_l1_loss(delta_bbox, target_delta_bbox)
            loss_test_sum += loss.cpu().detach().numpy()


            if i % 100 == 0:
                target_delta_bbox[:,0] *= 1088
                target_delta_bbox[:,1] *= 608
                target_delta_bbox[:,2] *= 1088
                target_delta_bbox[:,3] *= 608

                delta_bbox[:,0] *= 1088
                delta_bbox[:,1] *= 608
                delta_bbox[:,2] *= 1088
                delta_bbox[:,3] *= 608

                print()
                print('delta and target')
                print(delta_bbox)
                print(target_delta_bbox)
                print(loss)

        loss_test_mean = loss_test_sum / len(dataloader_test)
        # print(loss_test_mean)
        writer.add_scalar('test/loss', loss_test_mean, n_iter)

    writer.close()

    torch.save(gpn.state_dict(), opt.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=2, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--print-interval', type=int, default=40, help='print interval')
    parser.add_argument('--test-interval', type=int, default=9, help='test interval')
    parser.add_argument('--lr', type=float, default=1e-2, help='init lr')
    parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
    parser.add_argument('--network', type=str, default='alexnet', help='alexnet or resnet')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
    parser.add_argument('--save-path', type=str, default='model.pth', help='model path')
    parser.add_argument('--load-path', type=str, default='model.pth', help='path to load model')


    opt = parser.parse_args()

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt,
    )
