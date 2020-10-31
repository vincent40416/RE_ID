
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel
import os
import numpy as np
import argparse
from preprocessing import load_jpg
from model import PCBModel as Model
from model import weights_init_normal
from utils import adjust_lr_staircase


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('-r', '--run', type=int, default=1)
        parser.add_argument('--dataset', type=str, default='market1501',
                            choices=['market1501', 'cuhk03', 'duke', 'combined'])
        parser.add_argument('--trainset_part', type=str, default='trainval',
                            choices=['trainval', 'train'])

        parser.add_argument('--resize_h_w', type=eval, default=(384, 128))
        # These several only for training set
        parser.add_argument('--crop_prob', type=float, default=0)
        parser.add_argument('--crop_ratio', type=float, default=1)
        parser.add_argument('--batch_size', type=int, default=64)

        parser.add_argument('--steps_per_log', type=int, default=20)
        parser.add_argument('--epochs_per_val', type=int, default=1)

        parser.add_argument('--last_conv_stride', type=int, default=1, choices=[1, 2])
        # When the stride is changed to 1, we can compensate for the receptive field
        # using dilated convolution. However, experiments show dilated convolution is useless.
        parser.add_argument('--last_conv_dilation', type=int, default=1, choices=[1, 2])
        parser.add_argument('--num_stripes', type=int, default=6)
        parser.add_argument('--local_conv_out_channels', type=int, default=256)

        parser.add_argument('--exp_dir', type=str, default='')
        parser.add_argument('--model_weight_file', type=str, default='')

        parser.add_argument('--new_params_lr', type=float, default=0.1)
        parser.add_argument('--finetuned_params_lr', type=float, default=0.01)
        parser.add_argument('--staircase_decay_at_epochs',
                            type=eval, default=(41,))
        parser.add_argument('--staircase_decay_multiply_factor',
                            type=float, default=0.1)
        parser.add_argument('--total_epochs', type=int, default=60)

        args = parser.parse_args()

        # gpu ids
        # self.sys_device_ids = args.sys_device_ids

        # If you want to make your results exactly reproducible, you have
        # to fix a random seed.

        # The experiments can be run for several times and performances be averaged.
        # `run` starts from `1`, not `0`.
        self.run = args.run

        ###########
        # Dataset #
        ###########

        # If you want to make your results exactly reproducible, you have
        # to also set num of threads to 1 during training.

        self.dataset = args.dataset
        self.trainset_part = args.trainset_part

        # Image Processing

        # Just for training set
        self.crop_prob = args.crop_prob
        self.crop_ratio = args.crop_ratio
        self.resize_h_w = args.resize_h_w

        # Whether to scale by 1/255
        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        # self.train_mirror_type = 'random' if args.mirror else None
        self.train_batch_size = args.batch_size
        self.train_final_batch = False
        self.train_shuffle = True

        self.test_mirror_type = None
        self.test_batch_size = 32
        self.test_final_batch = True
        self.test_shuffle = False

        # dataset_kwargs = dict(
        #     name=self.dataset,
        #     resize_h_w=self.resize_h_w,
        #     scale=self.scale_im,
        #     im_mean=self.im_mean,
        #     im_std=self.im_std,
        #     batch_dims='NCHW',
        #     num_prefetch_threads=self.prefetch_threads)
        #
        # prng = np.random
        # if self.seed is not None:
        #     prng = np.random.RandomState(self.seed)
        # self.train_set_kwargs = dict(
        #     part=self.trainset_part,
        #     batch_size=self.train_batch_size,
        #     final_batch=self.train_final_batch,
        #     shuffle=self.train_shuffle,
        #     crop_prob=self.crop_prob,
        #     crop_ratio=self.crop_ratio,
        #     mirror_type=self.train_mirror_type,
        #     prng=prng)
        # self.train_set_kwargs.update(dataset_kwargs)

        # prng = np.random
        # if self.seed is not None:
        #     prng = np.random.RandomState(self.seed)
        # self.val_set_kwargs = dict(
        #     part='val',
        #     batch_size=self.test_batch_size,
        #     final_batch=self.test_final_batch,
        #     shuffle=self.test_shuffle,
        #     mirror_type=self.test_mirror_type,
        #     prng=prng)
        # self.val_set_kwargs.update(dataset_kwargs)

        # prng = np.random
        # if self.seed is not None:
        #     prng = np.random.RandomState(self.seed)
        # self.test_set_kwargs = dict(
        #     part='test',
        #     batch_size=self.test_batch_size,
        #     final_batch=self.test_final_batch,
        #     shuffle=self.test_shuffle,
        #     mirror_type=self.test_mirror_type,
        #     prng=prng)
        # self.test_set_kwargs.update(dataset_kwargs)

        ###############
        # ReID Model  #
        ###############

        # The last block of ResNet has stride 2. We can set the stride to 1 so that
        # the spatial resolution before global pooling is doubled.
        self.last_conv_stride = args.last_conv_stride
        # When the stride is changed to 1, we can compensate for the receptive field
        # using dilated convolution. However, experiments show dilated convolution is useless.
        self.last_conv_dilation = args.last_conv_dilation
        # Number of stripes (parts)
        self.num_stripes = args.num_stripes
        # Output channel of 1x1 conv
        self.local_conv_out_channels = args.local_conv_out_channels

        #############
        # Training  #
        #############

        self.momentum = 0.9
        self.weight_decay = 0.0005

        # Initial learning rate
        self.new_params_lr = args.new_params_lr
        self.finetuned_params_lr = args.finetuned_params_lr
        self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
        self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
        # Number of epochs to train
        self.total_epochs = args.total_epochs

        # How often (in epochs) to test on val set.
        self.epochs_per_val = args.epochs_per_val

        # How often (in batches) to log. If only need to log the average
        # information for each epoch, set this to a large value, e.g. 1e10.
        self.steps_per_log = args.steps_per_log

        # Only test and without training.
        # self.only_test = args.only_test
        #
        # self.resume = args.resume

        #######
        # Log #
        #######

        # If True,
        # 1) stdout and stderr will be redirected to file,
        # 2) training loss etc will be written to tensorboard,
        # 3) checkpoint will be saved
        # self.log_to_file = args.log_to_file

        # The root dir of logs.

        # Saving model weights and optimizer states, for resuming.
        # Just for loading a pretrained model; no optimizer states is needed.
        # self.model_weight_file = args.model_weight_file

# HR_train_data_path = 'C:/Users/User/Desktop/GAN_data_set/imagenet64_train/Imagenet64_train'


device_ids = [0, 1, 2, 3]
# batch size is in preprocessing
batch_size = 64
epochs = 0
num_epochs = 100
checkpoint_interval = 10
num_classes = 1502


def main():
    cfg = Config()

    ###################
    #  preprocessing  #
    ###################

    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    Market1501 = PATH + '/Market-1501-v15.09.15/gt_bbox/*.jpg'
    dataloader = load_jpg(Market1501, start_point=0, end_point=18000, train=True)
    test_data_set = load_jpg(Market1501, start_point=0, end_point=1000, shuffle=False)
    ###################
    #      Model      #
    ###################

    model = Model(
        last_conv_stride=cfg.last_conv_stride,
        num_stripes=cfg.num_stripes,
        local_conv_out_channels=cfg.local_conv_out_channels,
        num_classes=num_classes
    )
    model.cuda()
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # device_ids has a default : all
        model_w = DataParallel(model, device_ids=device_ids)
    model_w.to(device)

    if epochs != 0:
        model_w.load_state_dict(torch.load('./Model_log/RE_ID_%d.pth' % epochs))
    else:
        model_w.apply(weights_init_normal)
    #############################
    # Criteria and Optimizers   #
    #############################

    criterion = torch.nn.BCEWithLogitsLoss()
    finetuned_params = list(model.base.parameters())
    new_params = [p for n, p in model.named_parameters() if not n.startswith('base.')]

    param_groups = [{'params': finetuned_params, 'lr': cfg.finetuned_params_lr},
                    {'params': new_params, 'lr': cfg.new_params_lr}]
    optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.weight_decay, amsgrad=False)
    # optimizer = optim.SGD(
    #     param_groups,
    #     momentum=cfg.momentum,
    #     weight_decay=cfg.weight_decay)

    ##################
    # Resume Model   #
    ##################
    # if cfg.resume:
    #     # things to do
    #     print("resume")

    ##########
    #  Log   #
    ##########
    f = open("RE_ID_loss.txt", "w+")

    for epoch in range(epochs, num_epochs):
        # Epoch
        # Adjust Learning Rate
        adjust_lr_staircase(
            optimizer.param_groups,
            [cfg.finetuned_params_lr, cfg.new_params_lr],
            epoch + 1,
            cfg.staircase_decay_at_epochs,
            cfg.staircase_decay_multiply_factor)

        for i, img in enumerate(dataloader):
            if i == len(dataloader):
                break
            # model input
            imgs = Variable(img['img'].float()).cuda()
            label = Variable(img['label'].float()).cuda()

            _, logits_list = model_w(imgs)
            # label = label.permute(1, 0, 2)
            # print(np.shape(label))
            # print([criterion(logits, label) for logits in logits_list])

            loss = torch.sum(
                        torch.stack([criterion(logits, label) for logits in logits_list]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ############
            # Step Log #
            ############

            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]" %
                  (epoch, num_epochs, i, len(dataloader), loss))
            f.write("[Epoch %d/%d] [Batch %d/%d] [loss: %f]\n" %
                    (epoch, num_epochs, i, len(dataloader), loss))

        ##########
        #  TEST  #
        ##########
        total = 0
        correct = 0
        for i, test_img in enumerate(test_data_set):
            if i >= 5:
                print("accuracy:  %.6f" % (100 * correct/total))
                break
        #                     print(test_img)

            test_imgs = Variable(test_img['img'].float()).detach()
            test_label = Variable(test_img['label'].float()).detach()
            # test_label = test_label.permute(1, 0, 2)
            _, test_logits_list = model_w(test_imgs)

            # print(test_label.size(1))
            _, ind = torch.max(np.sum([logits for logits in test_logits_list], axis=0), 1)
            a = ind.cpu().numpy()

            test_label = test_label.numpy()
            # print(np.shape(test_label))
            indices = np.argmax(test_label, axis=1)
            # print(a)
            # print(indices)
            for j in range(len(a)):
                total += 1
                if a[j] == indices[j]:
                    correct += 1

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(model_w.state_dict(), './Model_log/RE_ID_%d.pth' % epoch)
    f.close()



# def Test():
#     g = open("FSR_PSNR.txt", "w+")
#     torch.cuda.empty_cache()
#     psnr_avg = 0
#     for i, test_img in enumerate(test_data_set):
#         if i > 10:
#             break
# #                     print(test_img)
#         print(i)
#         torch.cuda.empty_cache()
#         test_imgs_lr = Variable(input_lr.copy_(test_img['lr'])).detach()
#         test_imgs_pr = Variable(input_pr.copy_(test_img['pr'])).detach()
#         test_imgs_hr = Variable(input_hr.copy_(test_img['hr'])).detach()
#         output0, parsing_maps, test_fake_images, _, _, _, _ = srnet(test_imgs_lr, test_imgs_pr, test_imgs_hr)
#         psnr_val = psnr_cal(test_fake_images.data.cpu().numpy(), test_imgs_hr.data.cpu().numpy())
#         psnr_avg += psnr_val / 11
#         print("[PSNR: %f]" % psnr_val)
# #                     save_image(test_imgs_lr,"C:/Users/User/Desktop/Deep_Learning/GAN_images/TESTIMAGE'{0}'.png".format(i+(epoch)*batch_size), normalize=True)
#         save_image(torch.cat((test_fake_images.data, test_imgs_hr.data), -2),
#                    "./Saved_image/'{0}'.png".format(epochs), normalize=True)
#     print("PSNR_AVG: %f " % psnr_avg)
#     g.write("[Epoch %d/%d] [PSNR: %f]" % (epochs, num_epochs, psnr_val))
#     g.close()


# srnet=nn.DataParallel(srnet,device_ids=[0,1,2])
# if test == 1:
#     srnet = NetSR(loss = criterion_MSE,num_layers_res=2)
# else:

# Test()
if __name__ == '__main__':
    main()
