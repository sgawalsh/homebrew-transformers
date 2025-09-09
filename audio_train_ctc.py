import torch, trainer, os, audio_data, audio_model, settings, pickle
    

modelDict = {
    "ctcAudioFull": {"num_hiddens": 512, "num_blks": 6, "dropout": 0.1, "ffn_num_hiddens": 2048, "num_heads": 8},
    "ctcAudioFullNoDrop": {"num_hiddens": 512, "num_blks": 6, "dropout": 0.0, "ffn_num_hiddens": 2048, "num_heads": 8},
    "ctcAudioFullHalfDrop": {"num_hiddens": 512, "num_blks": 6, "dropout": 0.05, "ffn_num_hiddens": 2048, "num_heads": 8},
    "ctcAudioFull3": {"num_hiddens": 512, "num_blks": 3, "dropout": 0.1, "ffn_num_hiddens": 2048, "num_heads": 4},
    "ctcAudioFull3NoDrop": {"num_hiddens": 512, "num_blks": 3, "dropout": 0, "ffn_num_hiddens": 2048, "num_heads": 4},
    "ctcAudioHalf": {"num_hiddens": 256, "num_blks": 3, "dropout": 0.1, "ffn_num_hiddens": 1024, "num_heads": 4},
    "ctcAudioSmall": {"num_hiddens": 128, "num_blks": 2, "dropout": 0.1, "ffn_num_hiddens": 512, "num_heads": 2},
    "ctcAudioTest": {"num_hiddens": 256, "num_blks": 2, "dropout": 0.1, "ffn_num_hiddens": 2048, "num_heads": 2},
    }

def calcLoss_wav_2_letter(myData : audio_data.audio_data, myModel: audio_model.TransformerCTCEncoder, isTrain: bool):
    totalLoss = 0
    if isTrain:
        myModel.train()
        dataGen = myData.ctc_train_dataloader(mode)
    else:
        myModel.eval()
        dataGen = myData.ctc_val_dataloader(mode)

    toProcess = next(dataGen)

    for i, batch in enumerate(dataGen, 1):
        preds, predLengths, targLengths = [], torch.zeros(batch_size, dtype=torch.int32), torch.zeros(batch_size, dtype=torch.int32)
        for j, input in enumerate(batch[0]):
            pred = myModel(torch.unsqueeze(input.transpose(1, 0), 0).float())
            preds.append(pred)
            predLengths[j] = pred.shape[2]
            print(pred.argmax(1))
            if pred.isnan().any():
                print(input)
                print(pred)
                torch.save(myModel.state_dict(), f'{os.getcwd()}//models//{modelName}_nan_error')
                with open(f'{os.getcwd()}//error//errorData', 'wb+') as f:
                    pickle.dump({"input": input, "pred": pred}, f)
                raise Exception("NANS DETECTED")
            
        for j, targ in enumerate(batch[-1]):
            targLengths[j] = len(targ)

        if (predLengths < targLengths).any():
            continue

        predTensor = torch.zeros((batch_size, vocabSize, predLengths.max().item()))
        targTensor = torch.zeros((batch_size, targLengths.max().item()), dtype=torch.float32)

        # for j, targPred in enumerate(zip(batch[-1], preds)):
        #     targTensor[j][:targLengths[j]] = torch.tensor(targPred[0])
        #     predTensor[j][:predLengths[j]] = targPred[1]

        for j, targ in enumerate(batch[-1]):
            targTensor[j][:targLengths[j]] = torch.tensor(targ)

        for j, pred in enumerate(preds):
            predTensor[j][:predLengths[j]] = pred

        if decodePreds:
            for el in zip(predTensor, targTensor, targLengths):
                decodeString = audio_data.decodeCtc(el[0].transpose(1, 0).argmax(-1), myData.tgt_vocab.inv_data, mode)
                if any(c.isalpha() for c in decodeString):
                    print(decodeString)
                    print(audio_data.decodeTarg(el[1][:el[2]], myData.tgt_vocab.inv_data, mode))

        predTensor = torch.permute(predTensor, (2, 0, 1)) # BCT -> TBC

        # predTensor = [el.squeeze().T for el in preds] # -> T * C
        # predTensor = torch.nn.utils.rnn.pad_sequence(predTensor) # -> T * B * C

        loss = ctcLoss(predTensor, targTensor, predLengths, targLengths)

        if isTrain:
            with torch.no_grad():
                optim.zero_grad()
                loss.backward()
                optim.step()
            # for name, param in myModel.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient norm for {name}: {param.grad.norm()}")
        totalLoss += loss.item()
        print(f"Wavy {"Train" if isTrain else "Eval"} Epoch {epoch} - Loss: {totalLoss / i:.3f} - Processed {i * batch_size}/{toProcess}")

    return totalLoss / i

def calcLoss(myData : audio_data.audio_data, myModel: audio_model.TransformerCTCEncoder, isTrain: bool):
    totalLoss = 0
    if isTrain:
        myModel.train()
        dataGen = myData.ctc_train_dataloader(mode)
    else:
        myModel.eval()
        dataGen = myData.ctc_val_dataloader(mode)

    toProcess = next(dataGen)

    for i, batch in enumerate(dataGen, 1):
        preds, predLengths, targLengths = [], torch.zeros(batch_size, dtype=torch.int32), torch.zeros(batch_size, dtype=torch.int32)
        for j, input in enumerate(batch[0]):
            pred = myModel(torch.unsqueeze(input, 0).float(), torch.full((1,), input.shape[0]))
            preds.append(pred)
            predLengths[j] = pred.shape[1]
            # print(pred.argmax(-1))
            if pred.isnan().any():
                print(input)
                print(pred)
                torch.save(myModel.state_dict(), f'{os.getcwd()}//models//{modelName}_nan_error')
                with open(f'{os.getcwd()}//error//errorData', 'wb+') as f:
                    pickle.dump({"input": input, "pred": pred}, f)
                raise Exception("NANS DETECTED")
            
        for j, targ in enumerate(batch[-1]):
            targLengths[j] = len(targ)

        if (predLengths < targLengths).any():
            continue

        predTensor = torch.zeros((batch_size, predLengths.max().item(), vocabSize))
        targTensor = torch.zeros((batch_size, targLengths.max().item()), dtype=torch.int32)

        # for j, targPred in enumerate(zip(batch[-1], preds)):
        #     targTensor[j][:targLengths[j]] = torch.tensor(targPred[0])
        #     predTensor[j][:predLengths[j]] = targPred[1]

        for j, targ in enumerate(batch[-1]):
            targTensor[j][:targLengths[j]] = torch.tensor(targ)

        for j, pred in enumerate(preds):
            predTensor[j][:predLengths[j]] = pred

        if decodePreds:
            
            for el in zip(predTensor, targTensor, targLengths):
                decodeString = audio_data.decodeCtc(el[0].argmax(-1), myData.tgt_vocab.inv_data, mode)
                if any(c.isalpha() for c in decodeString):
                    print(decodeString)
                    print(audio_data.decodeTarg(el[1][:el[2]], myData.tgt_vocab.inv_data, mode))

        predTensor = predTensor.transpose(0,1) # BTC -> TBC

        loss = ctcLoss(predTensor, targTensor, predLengths, targLengths)

        if isTrain:
            with torch.no_grad():
                loss.backward()
                optim.step()
                optim.zero_grad()
        totalLoss += loss.item()
        # print([totalLoss, i])
        print(f"{mode} {"Train" if isTrain else "Eval"} Epoch {epoch} - Loss: {totalLoss / i:.3f} - Processed {i * batch_size}/{toProcess}")
        # print(loss.item())

    return totalLoss / i

device = settings.device
torch.set_default_device(device)
batch_size = 1
decodePreds = False

# import wav_2_letter
# mode = "letter"
# myData = audio_data.audio_data(batch_size=batch_size, isCtc = True, mode = mode)
# vocabSize = len(myData.tgt_vocab) # 0 reserved for blank character
# ctcLoss = torch.nn.CTCLoss()
# myTrainer = trainer.trainer(myData)

# myModel = wav_2_letter.wav2letter()
# optim =  torch.optim.Adam(myModel.parameters(), lr = 1e-6)
# # print(myModel)

# epochs = 10

# try:
#     myModel.load_state_dict(torch.load(f'{os.getcwd()}//models//wav2letter'))
# except:
#     pass

# trainLoss, evalLoss = [], []
# try:
#     for epoch in range(epochs):
#         trainLoss.append(calcLoss_wav_2_letter(myData, myModel, True))
#         with torch.no_grad():
#             evalLoss.append(calcLoss_wav_2_letter(myData, myModel, False))

#     torch.save(myModel.state_dict(), f'{os.getcwd()}//models//wav2letter')
# except Exception as e:
#     print(e)



combo = True
for mode in ["letter", "phoneme", "word"]:
    modelName = "ctcAudioFullNoDrop"
    params = modelDict[modelName]

    modelName += "_" + mode + "_combo" if combo else ""

    num_hiddens = params["num_hiddens"]
    num_blks = params["num_blks"]
    dropout = params["dropout"]
    ffn_num_hiddens = params["ffn_num_hiddens"]
    num_heads = params["num_heads"]

    myData = audio_data.audio_data(batch_size=batch_size, isCtc = True, mode = mode)
    vocabSize = len(myData.tgt_vocab) # 0 reserved for blank character
    ctcLoss = torch.nn.CTCLoss()
    myTrainer = trainer.trainer(myData)

    # myModel = audio_model.TransformerCTCEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, vocabSize)
    myModel = audio_model.comboModel(num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, vocabSize) if combo else audio_model.TransformerCTCEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, vocabSize)
    optim =  torch.optim.Adam(myModel.parameters(), lr = 1e-6)
    # print(myModel)

    epochs = 10

    try:
        myModel.load_state_dict(torch.load(f'{os.getcwd()}//models//{modelName}'))
    except:
        pass

    trainLoss, evalLoss = [], []
    try:
        for epoch in range(epochs):
            trainLoss.append(calcLoss(myData, myModel, True))
            with torch.no_grad():
                evalLoss.append(calcLoss(myData, myModel, False))

        torch.save(myModel.state_dict(), f'{os.getcwd()}//models//{modelName}')
    except Exception as e:
        print(e)
        continue