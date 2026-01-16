import os
import json
import sys
import shutil
import random
import pickle
import datetime
import argparse
import pathlib as path

import tqdm
import logging

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import torchvision  # type: ignore
from tensorboardX import SummaryWriter  # type: ignore

from model.utils import int_tuple, str_tuple, bool_flag
from model.metrics import iou,MetricAverage,image_acc,image_acc_ignore,binary_image_acc
from model.model import Model
from model.floorplan import FloorPlanDataset,floorplan_collate_fn
from model.loss import *
from model.box_utils import *
from model.utils import *

from ignite.contrib.handlers.tensorboard_logger import *  # type: ignore
from ignite.contrib.handlers import *  # type: ignore
from ignite.contrib.metrics import *  # type: ignore
from ignite.metrics.accuracy import _BaseClassification  # type: ignore
from ignite.engine import *  # type: ignore
from ignite.handlers import *  # type: ignore
from ignite.metrics import *  # type: ignore

def parse_args():
    parser = argparse.ArgumentParser()
    
    ''' Dataset '''
    parser.add_argument('--dataset_dir', default='./data', type=str)
    parser.add_argument('--image_size', default='128,128', type=int_tuple)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--with_house', default='0', type=bool_flag)
    parser.add_argument('--pos_dim', default=25, type=int)
    parser.add_argument('--area_dim', default=10, type=int)

    ''' Dataloader '''
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--train_shuffle', default='1', type=bool_flag)

    ''' Model '''
    # architecture
    parser.add_argument('--gene_layout', default='1', type=bool_flag)
    parser.add_argument('--box_refine', default='1', type=bool_flag)
    # input
    parser.add_argument('--embedding_dim', default=128,type=int)
    # refine
    parser.add_argument('--refinement_dims', default='1024, 512, 256, 128, 64',type=int_tuple)
    # box refine - now auto-detects num_objs from vocabulary
    parser.add_argument('--box_refine_arch', default=None,type=str)
    parser.add_argument('--roi_cat_feature',default='1',type=bool_flag)  
    # control
    parser.add_argument('--gt_box', default=0, type=bool_flag)
    parser.add_argument('--relative', default=1, type=bool_flag)

    ''' Loss '''
    parser.add_argument('--mutex', default=1, type=bool_flag)
    parser.add_argument('--inside', default=1, type=bool_flag)
    parser.add_argument('--coverage', default=1, type=bool_flag)
    parser.add_argument('--render', default=1, type=bool_flag)
    parser.add_argument('--nsample', default=100,type=int)
    parser.add_argument('--loss_refine', default=0, type=bool_flag)
    parser.add_argument('--render_refine', default=0, type=bool_flag)    
    
    ''' Optimizer '''
    parser.add_argument('--optimizer',default='Adam',type=str)
    parser.add_argument('--scheduler',default='plateau',type=str)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--decay_rate', default=1e-4, type=float)
    parser.add_argument('--step_size', default=10, type=float)
    parser.add_argument('--step_rate', default=0.5, type=float)

    ''' Checkpoints '''
    parser.add_argument('--save_interval', default=5, type=int)
    parser.add_argument('--n_saved', default=20, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--skip_train', default=0, type=bool_flag)

    ''' Trainer '''
    parser.add_argument('--seed', default=74269,type=int)
    parser.add_argument('--epoch', default=101,type=int)
    parser.add_argument('--start_epoch',default=None,type=int)

    ''' Others '''
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--multi_gpu', default=None, type=str)
    parser.add_argument('--suffix',default=None,type=str)
    parser.add_argument('--debug', default=0, type=bool_flag)
    parser.add_argument('--test', default=0, type=bool_flag)

    return parser.parse_args()

def check_manual_seed(args):
    seed = args.seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_model(args):
    return Model(embedding_dim=args.embedding_dim,
    image_size=args.image_size,
    input_dim = args.input_dim,
    attribute_dim=args.pos_dim+args.area_dim,
    refinement_dims=args.refinement_dims if args.gene_layout else None,    
    box_refine_arch=args.box_refine_arch if args.box_refine else None,
    roi_cat_feature=args.roi_cat_feature)

def get_dataset(args,split='valid'):
    return FloorPlanDataset(f'{args.dataset_dir}/data_{split}.mat')

def get_dataloader(args,dataset,split):
    print(f"{split},shuffle:",split=='train' and args.train_shuffle and (not args.debug))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if split=='train' and args.train_shuffle and (not args.debug) else False,
        num_workers=args.workers,
        drop_last=True if split=='train' else False,
        collate_fn=floorplan_collate_fn
        )

def  get_data_loaders(args):
    train_dataset = get_dataset(args,'train' if not args.debug else 'valid') if not args.skip_train else None
    valid_dataset = get_dataset(args,'valid')
    test_dataset = get_dataset(args,'test')
    
    train_loader = get_dataloader(args,train_dataset,'train') if not args.skip_train else None
    valid_loader = get_dataloader(args,valid_dataset,'valid')
    test_loader = get_dataloader(args,test_dataset,'test')
    return train_loader,valid_loader,test_loader

def get_optimizer(model,args):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = args.learning_rate,
            weight_decay=args.decay_rate
        )
    return optimizer

def get_scheduler(optimizer,args):
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_rate)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=args.step_rate,patience=args.step_size,threshold=0.005,verbose=True)
    return scheduler

def get_losses(args):
    loss = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight = torch.ones(18).to(device)
    weight[13]=weight[14]=0 # ignore External and ExteriorWall categories (not used in ResPlan)
    if args.gene_layout: 
        loss['gene_ce'] = torch.nn.CrossEntropyLoss(weight=weight)
    loss['box_mse'] = torch.nn.SmoothL1Loss()
    if args.box_refine:
        loss['box_ref_mse'] = torch.nn.SmoothL1Loss()
    if args.mutex:
        loss['mutex'] = MutexLoss(nsample=args.nsample)
    if args.inside:
        loss['inside'] = InsideLoss(nsample=args.nsample)
    if args.coverage:
        loss['coverage'] = CoverageLoss(nsample=args.nsample)
    if args.render:
        loss['render'] = BoxRenderLoss(nsample=args.nsample)
    return loss

def batch_cuda(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = list(batch)
    for i in range(len(batch)):
        if isinstance(batch[i],torch.Tensor):
            batch[i] = batch[i].to(device)
        elif isinstance(batch[i],list) and isinstance(batch[i][0],torch.Tensor):
            batch[i] = [e.to(device) for e in batch[i]]
    return batch

def main(args):
    args.epoch=args.epoch if not args.debug else 6
    print("Create dir...")
    start_date = str(datetime.datetime.now().strftime('%Y-%m-%d'))+("" if not args.debug else "_debug")+("" if not args.test else "_test")
    if not os.path.exists(f'../experiment'):
        os.mkdir(f'../experiment')
    experiment_dir = path.Path(f'../experiment/{start_date}')
    experiment_dir.mkdir(exist_ok=True)
    start_time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '' if args.suffix is None else args.suffix
    file_dir = path.Path(f'{experiment_dir}/DeepLayout_{start_time}')
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    shutil.copy(__file__,log_dir/'train.py')
    shutil.copytree('./model',log_dir/'model')
    output_dir = file_dir.joinpath('output/')
    output_dir.mkdir(exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir)+'/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if args.skip_train:
        logger.info(f'python {args.argv}')
    else: 
        logger.info(f'python {args.argv} --skip_train 1 --pretrain ')
    logger.info(args)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info(f'Use seed: {args.seed}')
    # check_manual_seed(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else args.multi_gpu

    print("Create dataloader...")
    train_loader,valid_loader,test_loader = get_data_loaders(args)
    print("Create model...")
    model = get_model(args)
    print("Gene:",model.refinement_net!=None and args.gene_layout)
    print("Refine:",args.box_refine)
    print("Cat feat:",args.roi_cat_feature)
    print("GT BOX:",args.gt_box)
    print("Iniside Loss:",args.inside)
    print("Coverage Loss:",args.coverage)
    print("Mutex Loss:",args.mutex)
    print("Render Loss:",args.render)
    logger.info(argparse.Namespace(embedding_dim=args.embedding_dim,
    image_size=args.image_size,
    input_dim = args.input_dim,
    attribute_dim=args.pos_dim+args.area_dim,
    refinement_dims=args.refinement_dims if args.gene_layout else None,    
    box_refine_arch=args.box_refine_arch if args.box_refine else None,
    roi_cat_feature=args.roi_cat_feature))
    logger.info(str(model))
    optimizer = get_optimizer(model,args)
    scheduler = get_scheduler(optimizer,args)
    loss = get_losses(args)

    if args.pretrain is not None:
        # Load checkpoint with CPU/GPU compatibility
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.pretrain} on {device}")

    # Move model to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    def update(engine,batch):
        model.train()
        
        # Learning rate warmup for first 3 epochs to prevent early divergence
        epoch = engine.state.epoch
        if epoch <= 3:
            warmup_factor = min(1.0, epoch / 3.0)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * warmup_factor
        
        optimizer.zero_grad()
        
        boundary,inside_box,objs,attrs,triples,layout,boxes,inside_coords,obj_to_img,triple_to_img,name = batch_cuda(batch)

        if args.relative: boxes = box_rel2abs(boxes,inside_box,obj_to_img)

        model_out = model(
            objs, 
            triples, 
            boundary,
            obj_to_img = obj_to_img,
            attributes=attrs,
            boxes_gt= boxes if args.gt_box else None, 
            generate = args.gene_layout and engine.state.epoch>1,
            refine = args.box_refine and engine.state.epoch>2,
            relative = args.relative,
            inside_box=inside_box if args.relative else None,
        )
        boxes_pred, gene_layout, boxes_refine = model_out
        
        # Initialize total_loss as None, will be set to first valid loss
        total_loss = None
        loss_items = {}
        epoch = engine.state.epoch
        step_weight = [0.1,0.5,1.0]
        for name in loss:
            l = None
            if name=='box_mse':
                l = loss[name](boxes_pred,boxes)
            else:
                if epoch>1:
                    if name=='gene_ce':
                        # Check for NaN in gene_layout before computing loss
                        if torch.isnan(gene_layout).any() or torch.isinf(gene_layout).any():
                            logging.error(f"NaN/Inf detected in gene_layout at epoch {epoch}")
                            continue
                        l = step_weight[epoch-2 if epoch<=3 else -1]*loss[name](gene_layout,layout)
                    elif name=='mutex':
                        l = 0.1*loss[name](boxes_pred,obj_to_img,objs)
                        if torch.isnan(l) or torch.isinf(l):
                            logging.warning(f"NaN/Inf in mutex loss at epoch {epoch}, skipping")
                            l = None
                        elif args.box_refine and args.loss_refine and epoch>2: 
                            l_refine = loss[name](boxes_refine,obj_to_img,objs)
                            if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
                                l += l_refine
                    elif name=='inside':
                        l = 0.1*loss[name](boxes_pred,inside_box,obj_to_img)
                        if torch.isnan(l) or torch.isinf(l):
                            logging.warning(f"NaN/Inf in inside loss at epoch {epoch}, skipping")
                            l = None
                        elif args.box_refine and args.loss_refine and epoch>2:
                            l_refine = loss[name](boxes_refine,inside_box,obj_to_img)
                            if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
                                l += l_refine
                    elif name=='coverage':
                        l = 0.1*loss[name](boxes_pred,inside_coords,obj_to_img)
                        if torch.isnan(l) or torch.isinf(l):
                            logging.warning(f"NaN/Inf in coverage loss at epoch {epoch}, skipping")
                            l = None
                        elif args.box_refine and args.loss_refine and epoch>2:
                            l_refine = loss[name](boxes_refine,inside_coords,obj_to_img)
                            if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
                                l += l_refine
                    elif name=='render':
                        l = loss[name](boxes_pred,boxes)
                        if torch.isnan(l) or torch.isinf(l):
                            logging.warning(f"NaN/Inf in render loss at epoch {epoch}, skipping")
                            l = None
                        elif args.box_refine and args.loss_refine and epoch>2:
                            l_refine = loss[name](boxes_refine,boxes)
                            if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
                                l += l_refine
                
                if epoch>2:
                    if name=='box_ref_mse':
                        l = step_weight[epoch-3 if epoch<=4 else -1]*loss[name](boxes_refine,boxes)
                        if torch.isnan(l) or torch.isinf(l):
                            logging.warning(f"NaN/Inf in box_ref_mse loss at epoch {epoch}, skipping")
                            l = None

            if l is not None:
                # Final safety check before adding to total
                if torch.isnan(l) or torch.isinf(l):
                    logging.warning(f"NaN/Inf detected in {name} loss, skipping")
                else:
                    if total_loss is None:
                        total_loss = l
                    else:
                        total_loss = total_loss + l
                    loss_items[name]=l.item()
        
        # If all losses were None/NaN, create a zero tensor for backward
        if total_loss is None:
            device = boxes_pred.device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            logging.warning(f"All losses were None/NaN at epoch {epoch}, using zero loss")
        
        loss_items['total_loss'] = total_loss.item()

        # Check for NaN before backward pass
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.error(f"NaN or Inf loss detected at epoch {epoch}, batch {engine.state.iteration}. Loss items: {loss_items}")
            logging.error(f"Skipping this batch to prevent gradient corruption.")
            return loss_items
        
        total_loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        return loss_items
    
    def inference(engine,batch):
        model.eval()
        with torch.no_grad():
            boundary,inside_box,objs,attrs,triples,layout,boxes,inside_coords,obj_to_img,triple_to_img,name = batch_cuda(batch)

            if args.relative: boxes = box_rel2abs(boxes,inside_box,obj_to_img)

            model_out = model(
                objs, 
                triples, 
                boundary,
                obj_to_img = obj_to_img,
                attributes=attrs,
                boxes_gt= boxes if args.gt_box else None, 
                generate = args.gene_layout,
                refine = args.box_refine,
                relative = args.relative,
                inside_box=inside_box if args.relative else None,
            )
            boxes_pred, gene_layout, boxes_refine = model_out
            
            # Initialize total_loss as None, will be set to first valid loss
            total_loss = None
            loss_items = {}
            for name in loss:
                l = None
                if name=='box_mse':
                    l = loss[name](boxes_pred,boxes)
                if engine.state.epoch>1:
                    if name=='gene_ce':
                        # Check for NaN in gene_layout before computing loss (validation)
                        if torch.isnan(gene_layout).any() or torch.isinf(gene_layout).any():
                            logging.warning(f"NaN/Inf detected in gene_layout during validation at epoch {engine.state.epoch}")
                            l = None
                        else:
                            l = loss[name](gene_layout,layout)
                    elif name=='mutex':
                        l = 0.1*loss[name](boxes_pred,obj_to_img,objs)
                        if torch.isnan(l) or torch.isinf(l):
                            l = None
                        elif args.box_refine and args.loss_refine: 
                            l_add = 0.1*loss[name](boxes_refine,obj_to_img,objs)
                            if not (torch.isnan(l_add) or torch.isinf(l_add)):
                                l += l_add
                    elif name=='inside':
                        l = 0.1*loss[name](boxes_pred,inside_box,obj_to_img)
                        if torch.isnan(l) or torch.isinf(l):
                            l = None
                        elif args.box_refine and args.loss_refine:
                            l_add = 0.1*loss[name](boxes_refine,inside_box,obj_to_img)
                            if not (torch.isnan(l_add) or torch.isinf(l_add)):
                                l += l_add
                    elif name=='coverage':
                        l = 0.1*loss[name](boxes_pred,inside_coords,obj_to_img)
                        if torch.isnan(l) or torch.isinf(l):
                            l = None
                        elif args.box_refine and args.loss_refine:
                            l_add = 0.1*loss[name](boxes_refine,inside_coords,obj_to_img)
                            if not (torch.isnan(l_add) or torch.isinf(l_add)):
                                l += l_add
                    elif name=='render':
                        l = loss[name](boxes_pred,boxes)
                        if torch.isnan(l) or torch.isinf(l):
                            l = None
                        elif args.box_refine and args.loss_refine:
                            l_add = loss[name](boxes_refine,boxes)
                            if not (torch.isnan(l_add) or torch.isinf(l_add)):
                                l += l_add

                if engine.state.epoch>2:
                    if name=='box_ref_mse':
                        l = loss[name](boxes_refine,boxes)
                        if torch.isnan(l) or torch.isinf(l):
                            l = None

                if l is not None:
                    # Safety check before adding (validation)
                    if not (torch.isnan(l) or torch.isinf(l)):
                        if total_loss is None:
                            total_loss = l
                        else:
                            total_loss = total_loss + l
                        loss_items[name]=l.item()
            
            # If all losses were None/NaN, create a zero tensor
            if total_loss is None:
                device = boxes_pred.device
                total_loss = torch.tensor(0.0, device=device)
                logging.warning(f"All losses were None/NaN during validation at epoch {engine.state.epoch}")
            
            # Final check for total_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                device = boxes_pred.device
                logging.warning(f"NaN/Inf total_loss during validation at epoch {engine.state.epoch}, setting to 0")
                total_loss = torch.tensor(0.0, device=device)
            
            loss_items['total_loss'] = total_loss.item()

            # boxes pred
            boxes_pred = boxes_pred.detach()
            boxes_pred = centers_to_extents(boxes_pred)

            if args.gene_layout:
                gene_layout = gene_layout*boundary[:,:1]

            # boxes refine
            if args.box_refine:
                boxes_refine = boxes_refine.detach()
                boxes_refine = centers_to_extents(boxes_refine)
                
            # gt
            boxes = centers_to_extents(boxes)

            return {
                'loss':loss_items,
                'pred':[
                    boxes_pred,
                    gene_layout.detach() if args.gene_layout else None,
                    boxes_refine if args.box_refine else None,
                    ],
                'gt':[layout,boxes]
            }
    
    print("Create trainer...")
    optimizer.step()
    scheduler.step(0)
    trainer = Engine(update)  # type: ignore
    valid_evaluator = Engine(inference)  # type: ignore

    if args.start_epoch is not None:
        @trainer.on(Events.STARTED)  # type: ignore
        def set_up_state(engine):
            engine.state.epoch = args.start_epoch

    total_func = lambda e:(e.state.metrics['box_iou']+(e.state.metrics['gene_acc'] if args.gene_layout else 0)+(e.state.metrics['box_refine_iou'] if args.box_refine else 0))

    @valid_evaluator.on(Events.COMPLETED)  # type: ignore
    def schedual(engine):
        optimizer.step()
        if args.scheduler == 'step':
            scheduler.step()
        else:
            scheduler.step(total_func(engine))

    @trainer.on(Events.EPOCH_COMPLETED)  # type: ignore
    def evaluate(engine):
        # Run validation only every 5 epochs to save time
        if engine.state.epoch % 5 == 0 or engine.state.epoch == 1:
            logging.info(f"Running validation at epoch {engine.state.epoch}")
            valid_evaluator.run(valid_loader)
            # Clear CUDA cache after validation to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logging.info(f"Validation completed, CUDA cache cleared")
        else:
            logging.info(f"Skipping validation at epoch {engine.state.epoch} (runs every 5 epochs)")

    # Metrics
    MetricAverage(output_transform=lambda output:iou(output['pred'][0],output['gt'][1])).attach(valid_evaluator,'box_iou')
    if args.gene_layout: 
        MetricAverage(output_transform=lambda output:image_acc_ignore(output['pred'][1],output['gt'][0],13)).attach(valid_evaluator,'gene_acc')
    if args.box_refine: 
        MetricAverage(output_transform=lambda output:iou(output['pred'][2],output['gt'][1])).attach(valid_evaluator,'box_refine_iou')
    
    metrics = ['img_acc','box_iou','mask_acc']

    # TQDM
    ProgressBar(persist=True).attach(trainer, output_transform=lambda o:{'loss':o['total_loss']}, metric_names='all')  # type: ignore
    ProgressBar(persist=False).attach(valid_evaluator, output_transform=lambda o:{'loss':o['loss']['total_loss']},metric_names='all')  # type: ignore
    
    # Tensorboard 
    tb_logger = TensorboardLogger(log_dir=log_dir)  # type: ignore
    tb_logger.attach(trainer,
                 log_handler=OutputHandler(tag="train",output_transform=lambda o: o,metric_names='all'),  # type: ignore
                 event_name=Events.ITERATION_COMPLETED)  # type: ignore
    tb_logger.attach(trainer,
                 log_handler=OptimizerParamsHandler(optimizer),  # type: ignore
                 event_name=Events.ITERATION_STARTED)  # type: ignore
    tb_logger.attach(valid_evaluator,
                 log_handler=OutputHandler(tag="valid",output_transform=lambda o:o['loss'],metric_names='all', global_step_transform=global_step_from_engine(trainer)),  # type: ignore
                 event_name=Events.EPOCH_COMPLETED)  # type: ignore
    
    # Logging
    @trainer.on(Events.EPOCH_COMPLETED)  # type: ignore
    def log_results(engine):
        logging.info(f'Train, Epoch{engine.state.epoch}, Loss: {str(engine.state.output)}')

    @valid_evaluator.on(Events.EPOCH_COMPLETED)  # type: ignore
    def log_results(engine):
        loss = engine.state.output['loss']
        metrics = engine.state.metrics
        logging.info(f'Valid, Epoch{engine.state.epoch}, Loss: {str(loss)}')
        logging.info(f'Valid, Epoch{engine.state.epoch}, Metrics: {str(metrics)}')
    
    # Checkpoint - save_interval moved to event handler attachment
    epoch_saver = ModelCheckpoint(checkpoints_dir, 'epoch', n_saved=args.n_saved, require_empty=False, create_dir=True)  # type: ignore
    latest_saver = ModelCheckpoint(checkpoints_dir, 'latest', score_function=lambda e:e.state.epoch, n_saved=1, require_empty=False, create_dir=True)  # type: ignore
    loss_saver = ModelCheckpoint(checkpoints_dir, 'loss', score_function=lambda e:-e.state.output['loss']['total_loss'], n_saved=1, require_empty=False, create_dir=True)  # type: ignore

    trainer.add_event_handler(Events.EPOCH_COMPLETED, latest_saver, {'model': model,'opt':optimizer})  # type: ignore
    # Use Events.EPOCH_COMPLETED(every=N) for save_interval
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=args.save_interval), epoch_saver, {'model': model,'opt':optimizer})  # type: ignore
    valid_evaluator.add_event_handler(Events.COMPLETED, loss_saver, {'model': model})  # type: ignore

    if not args.skip_train:
        trainer.run(train_loader,max_epochs=args.epoch)
    tb_logger.close()

    output = {}
    def test(engine,batch):
        model.eval()
        with torch.no_grad():
            boundary,inside_box,objs,attrs,triples,layout,boxes,inside_coords,obj_to_img,triple_to_img,name = batch_cuda(batch)

            model_out = model(
                objs, 
                triples, 
                boundary,
                obj_to_img = obj_to_img,
                attributes=attrs,
                boxes_gt= boxes if args.gt_box else None, 
                generate = args.gene_layout,
                refine = args.box_refine,
                relative = args.relative,
                inside_box=inside_box if args.relative else None,
            )
            boxes_pred, gene_layout, boxes_refine = model_out

            ''' box: x_c,y_c,w,h -> x0,y0,x1,y1 '''
            # boxes pred
            boxes_pred = boxes_pred.detach()
            boxes_pred = centers_to_extents(boxes_pred)
                
            # boxes refine
            if args.box_refine:
                boxes_refine = boxes_refine.detach()
                boxes_refine = centers_to_extents(boxes_refine)
                
            # gt
            if args.relative: boxes = box_rel2abs(boxes,inside_box,obj_to_img)
            boxes = centers_to_extents(boxes)

            ''' layout: B*C*H*W->B*H*W '''
            if args.gene_layout: 
                gene_layout = gene_layout*boundary[:,:1]
                gene_preds = torch.argmax(gene_layout.softmax(1).detach(),dim=1)
            
            ''' layout with outside'''
            for i in range(len(layout)):
                mask = boundary[i,0]==0
                if args.gene_layout: 
                    gene_preds[i][mask]=13

            ''' mertics '''
            # box iou
            box_ious = iou(boxes_pred,boxes)
            box_refine_ious = None
            if args.box_refine:
                box_refine_ious = iou(boxes_refine,boxes)

            gene_acc_all = None
            gene_acc_fg = None
            if args.gene_layout: 
                gene_acc_all = image_acc(gene_preds,layout)
                gene_acc_fg = image_acc_ignore(gene_preds,layout,13)

            ''' save output '''
            for i in range(len(layout)):
                ''' objs '''
                obj = objs[obj_to_img==i].cpu().numpy()

                ''' box '''
                box_pred = boxes_pred[obj_to_img==i]
                box_pred = box_pred.cpu().numpy()
                box_iou = box_ious[obj_to_img==i].view(-1).cpu().numpy()
                
                box_refine = None
                if args.box_refine:
                    box_refine = boxes_refine[obj_to_img==i].cpu().numpy()
                    box_refine_iou = box_refine_ious[obj_to_img==i].view(-1).cpu().numpy()

                ''' layout '''
                if args.gene_layout: 
                    gene_pred = gene_preds[i].cpu().numpy().astype('uint8')
  

                output[name[i]] = {
                        'obj':obj,
                        'box_gt':boxes[obj_to_img==i].cpu().numpy(),
                    
                        'box_pred':box_pred,
                        'box_iou':box_iou,

                        'box_refine':box_refine if args.box_refine else None,
                        'box_refine_iou':box_refine_iou if args.box_refine else None,

                        'gene_pred':gene_pred if args.gene_layout else None,
                        'gene_acc_all': gene_acc_all[i].item() if args.gene_layout else None,
                        'gene_acc_fg':gene_acc_fg[i].item() if args.gene_layout else None
                        }
            return {
                'pred':[
                    boxes_pred,#0
                    gene_preds if args.gene_layout else None,#1
                    boxes_refine if args.box_refine else None,#2
                    ],
                'gt':[layout,boxes]
            }                   
    
    test_evaluator = Engine(test)  # type: ignore

    MetricAverage(output_transform=lambda output:iou(output['pred'][0],output['gt'][1])).attach(test_evaluator,'box_iou')

    if args.gene_layout: 
        MetricAverage(output_transform=lambda output:image_acc_ignore(output['pred'][1],output['gt'][0],13)).attach(test_evaluator,'gene_acc')
        MetricAverage(output_transform=lambda output:image_acc(output['pred'][1],output['gt'][0])).attach(test_evaluator,'gene_acc_all')
    if args.box_refine: 
        MetricAverage(output_transform=lambda output:iou(output['pred'][2],output['gt'][1])).attach(test_evaluator,'box_refine_iou')

    ProgressBar(persist=False).attach(test_evaluator)  # type: ignore
    @test_evaluator.on(Events.COMPLETED)  # type: ignore
    def save_metrics(engine):
        metrics = engine.state.metrics
        with open(f'{output_dir}/output_{start_time}_metrics.json','w') as f:
            f.write(str(metrics))

    if not args.skip_train:
        test_evaluator.run(valid_loader)
    else:
        test_evaluator.run(test_loader)
    with open(f'{output_dir}/output_{start_time}.pkl','wb') as f:
        pickle.dump(output,f,pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()
    args.argv = ' '.join(sys.argv)
    main(args)
