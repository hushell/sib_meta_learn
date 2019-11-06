import torch
from tensorboardX import SummaryWriter
from utils.outils import progress_bar, AverageMeter, accuracy, getCi

class Algorithm:
    def __init__(self, args, netFeat, netSIB, optimizer, criterion):
        self.netFeat = netFeat
        self.netSIB = netSIB
        self.optimizer = optimizer
        self.criterion = criterion

        self.nbIter = args.nbIter
        self.nStep = args.nStep
        self.outDir = args.outDir
        self.nFeat = args.nFeat
        self.batchSize = args.batchSize
        self.inputH = args.inputH
        self.inputW = args.inputW
        self.nEpisode

        # Load pretrained model
        if args.resumeFeatPth :
            param = torch.load(args.resumeFeatPth)
            self.netFeat.load_state_dict(param)
            msg = 'Loading weight from {}'.format(args.resumeFeatPth)
            logger.info(msg)


    def validate(self, valLoader, mode='val', lr=None):
        if mode == 'test':
            print('\n\nTest mode: randomly sample {:d} episode for evaluation...'.format(self.nEpisode))
            nEpisode = self.nEpisode
        elif mode == 'val':
            print('\n\nValidation mode: evaluation pre-defined episodes...')
            nEpisode = len(valLoader)
            #valLoader = iter(valLoader)
        else:
            raise ValueError('mode is wrong!')

        episodeAccLog = []
        top1 = AverageMeter()

        self.netSIB.eval()

        #for batchIdx, data in enumerate(valLoader):
        for batchIdx in range(nEpisode):
            data = valLoader.getEpisode() if mode == 'test' else next(valLoader)

            SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                    data['SupportTensor'].squeeze(0), data['SupportLabel'].squeeze(0), \
                    data['QueryTensor'].squeeze(0), data['QueryLabel'].squeeze(0)

            with torch.no_grad():
                SupportFeat, QueryFeat = netFeat(SupportTensor), netFeat(QueryTensor)
                SupportFeat, QueryFeat, SupportLabel = \
                        SupportFeat.unsqueeze(0), QueryFeat.unsqueeze(0), SupportLabel.unsqueeze(0)

            if lr is None:
                lr = self.optimizer.param_groups[0]['lr']

            clsScore = self.netSIB(lr, SupportFeat, SupportLabel, QueryFeat)
            clsScore = clsScore.view(QueryFeat.size()[0] * QueryFeat.size()[1], -1)
            QueryLabel = QueryLabel.view(-1)
            acc1 = accuracy(clsScore, QueryLabel, topk=(1,))
            top1.update(acc1[0].item(), clsScore.size()[0])

            msg = 'Top1: {:.3f}%'.format(top1.avg)
            progress_bar(batchIdx, len(valLoader), msg)
            episodeAccLog.append(acc1[0].item())

        mean, ci95 = getCi(episodeAccLog)
        self.logger.info('Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95))
        return mean


    def train(self, trainLoader, valLoader, lr=None) :
        bestAcc = self.validate(valLoader, lr)
        self.logger.info('Acc improved over validation set from 0% ---> {:.3f}%'.format(bestAcc))

        self.netSIB.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        history = {'trainLoss' : [], 'trainAcc' : [], 'valAcc' : []}

        for episode in range(self.nbIter):
            data = trainLoader.getBatch()
            with torch.no_grad() :
                SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                        data['SupportTensor'], data['SupportLabel'], data['QueryTensor'], data['QueryLabel']

                SupportFeat, self.netFeat(SupportTensor.contiguous().view(-1, 3, self.inputW, self.inputH))
                QueryFeat = self.netFeat(QueryTensor.contiguous().view(-1, 3, self.inputW, self.inputH))

                SupportFeat, QueryFeat = SupportFeat.contiguous().view(self.batchSize, -1, self.nFeat), \
                        QueryFeat.view(self.batchSize, -1, self.nFeat)

            if lr is None:
                lr = self.optimizer.param_groups[0]['lr']

            self.optimizer.zero_grad()
            clsScore = self.netSIB(lr, SupportFeat, SupportLabel, QueryFeat)
            clsScore = clsScore.view(QueryFeat.size()[0] * QueryFeat.size()[1], -1)
            QueryLabel = QueryLabel.view(-1)
            loss = self.criterion(clsScore, QueryLabel)
            loss.backward()
            self.optimizer.step()

            acc1 = accuracy(clsScore, QueryLabel, topk=(1, ))
            top1.update(acc1[0].item(), clsScore.size()[0])
            losses.update(loss.item(), QueryFeat.size()[1])
            msg = 'Loss: {:.3f} | Top1: {:.3f}% '.format(losses.avg, top1.avg)
            progress_bar(episode, self.nbIter, msg)

            if episode % 1000 == 999 :
                acc = self.validate(valLoader, lr)

                if acc > bestAcc :
                    msg = 'Acc improved over validation set from {:.3f}% ---> {:.3f}%'.format(bestAcc , acc)
                    logger.info(msg)

                    bestAcc = acc
                    logger.info('Saving Best')
                    torch.save({
                                'lr': lr,
                                'netFeat': self.netFeat.state_dict(),
                                'SIB': self.netSIB.state_dict(),
                                'nbStep': self.nStep,
                                }, os.path.join(self.outDir, 'netSIBBest.pth'))

                logger.info('Saving Last')
                torch.save({
                            'lr': lr,
                            'netFeat': netFeat.state_dict(),
                            'SIB': netSIB.state_dict(),
                            'nbStep': self.nStep,
                            }, os.path.join(self.outDir, 'netSIBLast.pth'))

                msg = 'Iter {:d}, Train Loss {:.3f}, Train Acc {:.3f}%, Val Acc {:.3f}%'.format(
                        episode, losses.avg, top1.avg, acc)
                logger.info(msg)
                history['trainLoss'].append(losses.avg)
                history['trainAcc'].append(top1.avg)
                history['valAcc'].append(acc)

                losses = AverageMeter()
                top1 = AverageMeter()

        return bestAcc, acc, history
