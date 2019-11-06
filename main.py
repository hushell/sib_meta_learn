import torch
import random
import itertools
import json
import os

from algorithm import Algorithm
from networks import get_featnet
from sib import ClassifierSIB
from dataset import dataset_setting
from dataloader import TrainSampler, ValLoader, EpisodeSampler
from utils.config import get_config
from utils.utils import get_logger, set_random_seed

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

#############################################################################################
## hyper-parameters
args = get_config()

# logging to the file and stdout
logger = get_logger(args.logDir, args.expName)

# fix random seed to reproduce results
set_random_seed(args.seed)
logger.info('Random seed: {:d}'.format(args.seed))

#############################################################################################
## datasets
trainTransform, valTransform, inputW, inputH, \
        trainDir, valDir, testDir, episodeJson, nbCls = \
        dataset_setting(args.dataset, args.nSupport)

args.inputW = inputW
args.inputH = inputH

trainLoader = TrainSampler(imgDir = trainDir,
                           nClsEpisode = args.nClsEpisode,
                           nSupport = args.nSupport,
                           nQuery = args.nQuery,
                           transform = trainTransform,
                           useGPU = args.cuda,
                           inputW = inputW,
                           inputH = inputH,
                           batchSize = args.batchSize)

valLoader = ValLoader(episodeJson,
                      valDir,
                      inputW,
                      inputH,
                      valTransform,
                      args.cuda)

testLoader = EpisodeSampler(imgDir = testDir,
                            nClsEpisode = args.nClsEpisode,
                            nSupport = args.nSupport,
                            nQuery = args.nQuery,
                            transform = valTransform,
                            useGPU = args.cuda,
                            inputW = inputW,
                            inputH = inputH)


#############################################################################################
## Networks
netFeat = get_featnet(args.architecture, inputW, inputH)
netSIB = ClassifierSIB(args.nClsEpisode, netFeat.nFeat, args.nStep)

## Optimizer
optimizer = torch.optim.SGD(itertools.chain(*[netSIB.parameters(),]),
                                             args.lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weightDecay,
                                             nesterov=True)

## Loss
criterion = nn.CrossEntropyLoss()

## Algorithm class
alg = Algorithm(args, netFeat, netSIB, optimizer, criterion)


#############################################################################################
## main loop
if args.mode == 'train':
    bestAcc, lastAcc, history = alg.train(trainLoader, valLoader)

    ## Finish training!!!
    msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netSIBBest.pth'),
                            os.path.join(args.outDir, 'netSIBBest{:.3f}.pth'.format(bestAcc)))
    logger.info(msg)
    os.system(msg)

    msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netSIBLast.pth'),
                            os.path.join(args.outDir, 'netSIBLast{:.3f}.pth'.format(lastAcc)))
    logger.info(msg)
    os.system(msg)

    with open(os.path.join(args.outDir, 'history.json'), 'w') as f :
        json.dump(history, f)

    msg = 'mv {} {}'.format(args.outDir, '{}_{:.3f}'.format(args.outDir, bestAcc))
    logger.info(msg)
    os.system(msg)

elif args.mode == 'test':
    mean, ci95 = alg.test(test_loader)
    logger.info('Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95))

