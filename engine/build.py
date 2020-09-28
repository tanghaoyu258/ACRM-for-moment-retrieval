import os
import data
import json
import torch
import solver
import modeling
import numpy as np

from torch.nn.init import xavier_normal_, normal_
from utils.visualization import Visualization
from utils.miscellaneous import mkdir

from tensorboardX import SummaryWriter

def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 20))
    print("learning rate %f in epoch %d" % (lr, epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def trainer(cfg):
    print('trainer')
    dataloader_train, dataset_size_train = data.make_dataloader(cfg, is_train=True)
    dataloader_test, dataset_size_test   = data.make_dataloader(cfg, is_train=False)
    print(dataset_size_train)
    print(dataset_size_test)

    model = modeling.build(cfg)
    if cfg.MODE_TRAIN == 'resume':
        model = torch.load("./checkpoints/{}/model_{}_epoch_{}".format(cfg.EXPERIMENT_NAME, cfg.MODEL_NAME, cfg.MODE_TRAIN_RESUME_EPOCH))
    
    model = torch.load("./checkpoints/{}/model_{}_epoch_{}".format(cfg.EXPERIMENT_NAME, cfg.MODEL_NAME, 6))

    model.cuda()
    optimizer = solver.make_optimizer(cfg, model)
    #model = torch.load("/home/crodriguezo/projects/phd/moment-localization-with-NLP/mlnlp_lastversion/checkpoints/anet_config7/model_epoch_80")
    

    vis_train = Visualization(cfg, dataset_size_train)
    vis_test  = Visualization(cfg, dataset_size_test, is_train=False)

    writer_path = os.path.join(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
    writer = SummaryWriter(writer_path)

    total_iterations = 0
    total_iterations_val = 0
    cfg.EPOCHS =1
    for epoch in range(cfg.EPOCHS):
        model.eval()
        sumloss = 0
        sumsample = 0
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader_test):
                index     = batch[0]

                videoFeat = batch[1].cuda()
                videoFeat_lengths = batch[2].cuda()

                tokens         = batch[3].cuda()
                tokens_lengths = batch[4].cuda()
                if cfg.MODEL_NAME == 'TMLGA':
                    start    = batch[5].cuda()
                    end      = batch[6].cuda()
                    localiz  = batch[7].cuda()
                    frame_start = batch[13]
                    frame_end = batch[14] 
                else:
                    start    = batch[5]
                    end      = batch[6]
                    localiz  = batch[7]
                    frame_start = batch[13].cuda()
                    frame_end = batch[14].cuda()

                localiz_lengths = batch[8]
                time_starts = batch[9]
                time_ends = batch[10]
                factors = batch[11]
                fps = batch[12]

                duration = batch[15]
                vid_names = batch[16]
                loss, individual_loss, pred_start, pred_end, attention,atten_loss = model(videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end, localiz,frame_start,frame_end)
                sumloss += loss.item()* float(videoFeat.shape[0])
                sumsample += videoFeat.shape[0]
                # print("Test_Loss :{}".format(loss))
                vis_test.run(index, pred_start, pred_end, start, end, videoFeat_lengths, epoch, loss.detach(), individual_loss, \
                    attention,atten_loss, time_starts, time_ends, factors, fps, duration,vid_names)
                #print(loss)
                writer.add_scalar(
                    f'mlnlp/Progress_Valid_Loss',
                    loss.item(),
                    total_iterations_val)

                writer.add_scalar(
                    f'mlnlp/Progress_Valid_Atten_Loss',
                    atten_loss.item(),
                    total_iterations_val)

                writer.add_scalar(
                    f'mlnlp/Progress_Valid_Mean_IoU',
                    vis_test.mIoU[-1],
                    total_iterations_val)

                total_iterations_val += 1
                # del videoFeat,videoFeat_lengths,tokens,tokens_lengths,start,end,localiz
                # torch.cuda.empty_cache()
        print("Test_Loss :{}".format(sumloss/sumsample))
        writer.add_scalar(
            f'mlnlp/Valid_Loss',
            np.mean(vis_test.loss),
            epoch)

        writer.add_scalar(
            f'mlnlp/Valid_Mean_IoU',
            np.mean(vis_test.mIoU),
            epoch)

        a = vis_test.plot(epoch)
        writer.add_scalars(f'mlnlp/Valid_tIoU_th', a, epoch)



def tester(cfg):
    print('testing')
    dataloader_test, dataset_size_test   = data.make_dataloader(cfg, is_train=False)

    model = modeling.build(cfg)

    if cfg.TEST.MODEL.startswith('.'):
        load_path = cfg.TEST.MODEL.replace(".", os.path.realpath("."))
    else:
        load_path = cfg.TEST.MODEL

    model = torch.load(load_path)
    model.cuda()

    vis_test  = Visualization(cfg, dataset_size_test, is_train=False)

    writer_path = os.path.join(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
    writer = SummaryWriter(writer_path)

    total_iterations = 0
    total_iterations_val = 0

    model.eval()
    epoch = 1
    for iteration, batch in enumerate(dataloader_test):
        index     = batch[0]

        videoFeat = batch[1].cuda()
        videoFeat_lengths = batch[2].cuda()

        tokens         = batch[3].cuda()
        tokens_lengths = batch[4].cuda()

        start    = batch[5].cuda()
        end      = batch[6].cuda()
        
        localiz  = batch[7].cuda()
        localiz_lengths = batch[8]
        
        time_starts = batch[9]
        time_ends = batch[10]
        
        factors = batch[11]
        fps = batch[12]
        frame_start = batch[13]
        frame_end =batch[14]

        loss, individual_loss, pred_start, pred_end, attention, atten_loss = model(videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end, localiz,frame_start,frame_end)
        aux = vis_test.run(index, pred_start, pred_end, start, end, videoFeat_lengths, epoch, loss.detach(), individual_loss, attention, atten_loss, time_starts, time_ends, factors, fps)
        total_iterations_val += 1
    a = vis_test.plot(epoch)


