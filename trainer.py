import os, pickle, torch, sys, datetime
from torch.nn import functional as F
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from time import time
from torch.utils.tensorboard import SummaryWriter

class trainer:
    def __init__(self, myData):

        self.data = myData
        self.bestLoss = float('inf')
        self.bestBleu = float('-inf')
        self.bleuWeights = {1: (1, 0, 0, 0), 2: (.5, .5, 0, 0), 3: (.33, .33, .33, 0), 4: (.25, .25, .25, .25)}
        self.evalDataloader, self.trainDataloader = None, None

    def pickleData(self):
        for batch in self.data.train_dataloader():
            src = self.data.src_vocab.to_tokens(batch[0][0])
            input = self.data.tgt_vocab.to_tokens(batch[1][0])
            tgt = self.data.tgt_vocab.to_tokens(batch[3][0])

            print(src, input, tgt)
            with open(f'{os.getcwd()}//testData//2', 'wb+') as f:
                pickle.dump(batch, f)
            break

    def fit(self, model, lr, epochs = 1, showTranslations = False, calcBleu = True, loadModel = False, shutDown = False, modelName = "myModel", bleuPriority = True, fromBest = True):
        self.model = model
        self.model.decoder.predictMode = False

        self.optim = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min', .5, 10)
        self.showTranslations = showTranslations
        self.modelName = modelName
        self.calcBleu = calcBleu

        trainLoss, evalLoss, trainBleu, evalBleu = [], [], [], []
        self.i = 0
        self.epochs = epochs

        self.writer = SummaryWriter(log_dir = f'runs/encoderDecoder/{modelName}')

        checkpoint_path = f'checkpoints/{modelName}_state.pth'
        
        if loadModel:
            if fromBest:
                try:
                    self.model.load_state_dict(torch.load(f'{os.getcwd()}//models//{modelName}'))
                    if not self.evalDataloader:
                        print("Generate eval dataloader")
                        self.evalDataloader = self.data.val_dataloader()
                    self.bestLoss, self.bestBleu = self._train_cycle(True)
                except FileNotFoundError:
                    print("No file available to be loaded")
            else:
                if os.path.exists(checkpoint_path):
                    state = torch.load(checkpoint_path)
                    self.i = state['epoch']
                    self.model.load_state_dict(state['model_state'])
                    self.optim.load_state_dict(state['optimizer_state'])
                    trainLoss = state['train_loss']
                    trainBleu = state['train_bleu']
                    evalLoss = state['eval_loss']
                    evalBleu = state['eval_bleu']
                    self.bestLoss = state['best_loss']
                    self.bestBleu = state['best_bleu']
                    print(f'Checkpoint found, resumed from epoch {self.i}')
                    self.epochs += self.i
                else:
                    print("No checkpoint found")

        if not self.evalDataloader:
            print("Generate eval dataloader")
            self.evalDataloader = self.data.val_dataloader()
        if not self.trainDataloader:
            print("Generate train dataloader")
            self.trainDataloader = self.data.train_dataloader()
        for _ in range(epochs):
            self.i += 1
            cycleLoss, bleuScore = self._train_cycle()
            trainLoss.append(cycleLoss)
            trainBleu.append(bleuScore)
            self.writer.add_scalar('Loss/Train', cycleLoss, self.i)
            self.writer.add_scalar('BLEU/Train', bleuScore, self.i)
            print('Done training')

            cycleLoss, bleuScore = self._train_cycle(True)
            evalLoss.append(cycleLoss)
            evalBleu.append(bleuScore)
            self.writer.add_scalar('Loss/Eval', cycleLoss, self.i)
            self.writer.add_scalar('BLEU/Eval', bleuScore, self.i)
            if bleuPriority:
                if bleuScore > self.bestBleu:
                    self.bestBleu = bleuScore
                    torch.save(self.model.state_dict(), f'{os.getcwd()}//models//{modelName}')
            elif cycleLoss < self.bestLoss:
                self.bestLoss = cycleLoss
                torch.save(self.model.state_dict(), f'{os.getcwd()}//models//{modelName}')
            print('Done eval')

            # save checkpoint
            state = {
                'epoch': self.i,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optim.state_dict(),
                'train_loss': trainLoss,
                'train_bleu': trainBleu,
                'eval_loss': evalLoss,
                'eval_bleu': evalBleu,
                'best_loss': self.bestLoss,
                'best_bleu': self.bestBleu
            }
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(state, checkpoint_path)

        if shutDown:
            os.system('shutdown -s')

        fig, axes = plt.subplots(2, 1)
        axes[0].plot(range(1, len(trainLoss) + 1), trainLoss, label='Training')
        axes[0].plot(range(1, len(evalLoss) + 1), evalLoss, label='Eval')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[1].plot(range(1, len(trainBleu) + 1), trainBleu, label='Training')
        axes[1].plot(range(1, len(evalBleu) + 1), evalBleu, label='Eval')
        axes[1].set_title('Bleu Score')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Bleu Score')
        axes[1].legend()

        fig.tight_layout(pad=5.0)

        plt.savefig(f'plots/{modelName}_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}')


    def eval_cycle(self, model, modelName, showTranslations = True, calcBleu = True):
        self.model = model
        self.model.decoder.predictMode = False
        try:
            self.model.load_state_dict(torch.load(f'{os.getcwd()}//models//{modelName}'))
        except FileNotFoundError:
            print("No file available to be loaded")

        self.i, self.epochs = 1, 1
        self.showTranslations = showTranslations
        self.calcBleu = calcBleu

        return self._train_cycle(True)


    def _show_translations(self, inSrc, inPred, inTarg):
        refs, cans = [], []
        for line in zip(inSrc, inPred, inTarg):
            src, pred, targ = [], [], []
            if hasattr(self.data, 'src_vocab'):
                for token in line[0]:
                    if token == self.data.src_vocab['<eos>']:
                        break
                    src.append(self.data.src_vocab.to_token(token.item()))
            for token in line[1]:
                if token == self.data.tgt_vocab['<eos>']:
                    break
                pred.append(self.data.tgt_vocab.to_token(token.item()))
            for token in line[2]:
                if token == self.data.tgt_vocab['<eos>']:
                    break
                targ.append(self.data.tgt_vocab.to_token(token.item()))

            print(f'{f'src: {" ".join(src)}' if src else ""}\npred: {" ".join(pred)}\ntarg: {" ".join(targ)}\n')
            refs.append(targ)
            cans.append(pred)
    
    def _trim_eos(self, ref, can):
        eos = self.data.tgt_vocab['<eos>']

        try:
            ref = ref[:ref.index(eos)]
        except ValueError:
            pass
        try:
            can = can[:can.index(eos)]
        except ValueError:
            pass
        return ref, can

    def _train_cycle(self, isEval = False):
        if isEval:
            dataGen = self.evalDataloader if self.evalDataloader else self.data.val_dataloader()
            self.model.eval()
            dataLength = self.data.valDataLength
        else:
            dataGen = self.trainDataloader if self.trainDataloader else self.data.train_dataloader()
            self.model.train()
            dataLength = self.data.trainDataLength

        runningLoss, totalBleuScore = 0.0, 0.0
        start = time()
        for i, data in enumerate(dataGen, 1):

            Y = self.model(*data[:-1]) # (srcTensor, tgtTensor t-1, srcValidLens)
            loss = self._loss(Y, data[-1]) # (tgtTensor)

            if self.showTranslations:
                self._show_translations(data[0], torch.argmax(Y, 2), data[3])
            
            if self.calcBleu:
                bleuScores = 0.0
                for j, (ref, can) in enumerate(zip(data[3].tolist(), torch.argmax(Y, 2).tolist()), 1):
                    ref, can = self._trim_eos(ref, can)
                    try:
                        bleuScores += sentence_bleu([ref], can, weights=self.bleuWeights[min(4, len(can))])
                    except KeyError:
                        pass
                
                totalBleuScore += bleuScores / j

            if not isEval:
                with torch.no_grad():
                    loss.backward()
                    self.optim.step()
                    self.scheduler.step(loss)
                    self.optim.zero_grad()
                    
            runningLoss += loss.item() * self.data.batch_size
            try:
                print(f'{self.i}/{self.epochs} - {i * self.data.batch_size}/{dataLength} - {100 * i * self.data.batch_size/dataLength:.3f}% - Running Loss: {runningLoss / i:.3f} - Loss: {loss.item():.3f} - Bleu: {totalBleuScore / i * 100:.2f} - Speed: {(i * self.data.batch_size / (time() - start)):.3f}') # - Speed: {i * self.data.batch_size / (time() - start):.3f}
            except ZeroDivisionError:
                pass


        return runningLoss / self.data.batch_size / i, totalBleuScore / i

    def _loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = torch.reshape(Y, (-1,))
        return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none', label_smoothing=0.1)