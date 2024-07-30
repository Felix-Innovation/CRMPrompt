from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import *
from torchvision import transforms
import numpy as np

import clip
from models import image_prompters, text_prompters, multimodal_prompter

from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname

from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--image_learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument('--text_learning_rate', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--multimodal_learning_rate', type=float, default=1,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='ViT-B/16')
    parser.add_argument('--prompt_size', type=int, default=32,
                        help='size for visual prompts')

    # dataset
    parser.add_argument('--root', type=str,
                        default=r'./dataset',
                        help='dataset')

    parser.add_argument('--dataset', type=str, default='DTD',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')


    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')

    #################################

    parser.add_argument('--prefix_length', type=int, default=8,
                        help='number of prefix tokens')
    parser.add_argument('--suffix_length', type=int, default=8,
                        help='number of suffix tokens')
    parser.add_argument('--n_shot', type=int, default=16,
                        help='number of shots for few-shot learning')



    args = parser.parse_args()

    args.filename = '_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}'. \
        format(args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.image_learning_rate, args.text_learning_rate, args.weight_decay, args.batch_size,
               args.warmup, args.trial)

    return args

class DatasetLoader:
    def __init__(self, data_dir, batch_size, num_workers, dataset_name, preprocess, n_shot):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name.lower()
        self.preprocess = preprocess
        self.n_shot = n_shot  # few_shot

    def load_data(self):
        #if self.dataset_name in ['dtd']:
        if self.dataset_name == 'dtd':
            train_dataset = DTD(root=self.data_dir, split='train', transform=self.preprocess, download=True)
            test_dataset = DTD(root=self.data_dir, split='test', transform=self.preprocess, download=True)

            # train_dataset = EuroSAT(root=self.data_dir, split='train', transform=self.preprocess, download=True)
            # test_dataset = EuroSAT(root=self.data_dir, split='test', transform=self.preprocess, download=True)
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported")

        train_indices = self.get_few_shot_indices(train_dataset, self.n_shot)
        train_dataset = Subset(train_dataset, train_indices)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True,
                                  num_workers=self.num_workers, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers,
                                shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True,
                                 num_workers=self.num_workers, shuffle=False)

        #class_names = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
        #class_names = full_dataset.classes
        class_names = [str(i) for i in range(102)]
        return train_loader, val_loader, test_loader, class_names

    def get_few_shot_indices(self, dataset, num_samples_per_class):
        try:
            targets = np.array([y for _, y in dataset])
        except:
            print("check root path")
            return []

        indices = []
        for class_index in np.unique(targets):
            class_indices = np.where(targets == class_index)[0]
            np.random.shuffle(class_indices)
            selected_indices = class_indices[:num_samples_per_class]
            indices.extend(selected_indices)
        return indices

best_acc1 = 0

def main():
    global best_acc1, device

    args = parse_option()
    args.device = device
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    model, preprocess = clip.load('RN50', device, jit=False)
    convert_models_to_fp32(model)
    model.eval()

    # create data
    template = 'This is a photo of a {},a type of texture.'
    print(f'template: {template}')

    data_loader = DatasetLoader(data_dir=args.root, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                dataset_name=args.dataset, preprocess=preprocess, n_shot=args.n_shot)

    train_loader, val_loader, test_loader, class_names = data_loader.load_data()
    texts = class_names

    args.class_num = len(class_names)

    # create model
    mm_prompter = multimodal_prompter.MultimodalPromptLearner(args).to(device)

    # define criterion and optimizer
    optimizer_multimodal = torch.optim.SGD(mm_prompter.parameters(),
                                           lr=args.multimodal_learning_rate,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    total_steps = len(train_loader) * 100
    scheduler_multimodal = cosine_lr(optimizer_multimodal, args.multimodal_learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)


    epochs_since_improvement = 0

    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, texts, model, mm_prompter, optimizer_multimodal, scheduler_multimodal, criterion, scaler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, texts, model, mm_prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

    wandb.run.finish()

def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id

def get_prompt(args, model, texts, images, mm_prompter):
    actionlist = np.array(texts)
    actiontoken = np.array([convert_to_token(a) for a in actionlist])

    with torch.no_grad():
        actionembed = model.encode_text_light(torch.tensor(actiontoken).to(device))

    actiondict = OrderedDict((texts[i], actionembed[i].cpu().data.numpy()) for i in range(args.class_num))
    actiontoken = OrderedDict((texts[i], actiontoken[i]) for i in range(args.class_num))

    image_features, text_embedding, classname_tokens = mm_prompter(images, texts, actiondict, actiontoken)
    text_features = model.encode_text(text_embedding, classname_tokens)

    return image_features, text_features

def train(train_loader, texts, model, mm_prompter, optimizer_multimodal, scheduler_multimodal, criterion, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    mm_prompter.train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate  调整学习率
        step = num_batches_per_epoch * epoch + i   #计算当前的步数
        # 调整学习率
        scheduler_multimodal(step)
        # 梯度清零
        optimizer_multimodal.zero_grad()

        images = images.to(device)
        target = target.to(device)

        # with automatic mixed precision  自动混合精度训练（可提升训练速度和节省内存）
        with autocast():
            image_features, text_features = get_prompt(args, model, texts, images, mm_prompter)

            output, _ = model(image_features, text_features, text_emb=True)
            loss = criterion(output, target)
            scaler.scale(loss).backward()

            scaler.step(optimizer_multimodal)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # 显示进度
        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg
                })

    return losses.avg, top1.avg

def validate(val_loader, texts, model, mm_prompter, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_org, top1_prompt],
        prefix='Validate: ')

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)

            # origin
            template = 'This is a photo of a {},a type of texture.'
            hand_craft_prompt = [template.format(label) for label in texts]
            org_text_tokens = clip.tokenize(hand_craft_prompt).to(device)

            # with prompt  用提示生成器生成的提示
            image_features, text_features = get_prompt(args, model, texts, images, mm_prompter)

            # compute output  计算输出和损失
            output_prompt, _ = model(image_features, text_features, text_emb=True)
            output_org, _ = model(images, org_text_tokens)
            loss = criterion(output_prompt, target)

            # measure accuracy and record loss  计算准确率并记录损失
            acc1 = accuracy(output_prompt, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_prompt.update(acc1[0].item(), images.size(0))

            acc1 = accuracy(output_org, target, topk=(1,))
            top1_org.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))

        if args.use_wandb:
            wandb.log({
                'val_loss': losses.avg,
                'val_acc_prompt': top1_prompt.avg,
                'val_acc_org': top1_org.avg,
            })

    return top1_prompt.avg

if __name__ == '__main__':
    main()
