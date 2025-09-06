import torch, trainer, os, model, data, settings, pickle, decode, matplotlib.pyplot as plt
from settings import modelDict, modelName


def compare_models(modelList, fileName = "results.pkl"):
    results = {}

    for modelName in modelList:
        myModel, myData, myTrainer = load_model_data_trainer(modelName)
        del(myData)
        score = myTrainer.eval_cycle(myModel, modelName, showTranslations=False, calcBleu=True)

        results[modelName] = score

    with open(f'{os.getcwd()}//{fileName}', 'wb+') as f:
        pickle.dump(results, f)

def show_results(fileName = "results.pkl"):

    with open(f'{os.getcwd()}//{fileName}', 'rb') as f:
        results = pickle.load(f)

    values = results.values()

    plt.subplot(2, 1, 1)
    plt.bar(results.keys(), [x[0] for x in values])
    plt.xlabel("Model")
    plt.ylabel("Loss")
    plt.title("Model Losses")
    plt.subplot(2, 1, 2)
    plt.bar(results.keys(), [x[1] for x in values])
    plt.xlabel("Model")
    plt.ylabel("Bleu Score")
    plt.title("Bleu Scores")
    plt.show()
    

def load_model_data_trainer(modelName):
    params = modelDict[modelName]
    myData = data.europarl_data(batchSize)

    encoder = model.TransformerEncoder(len(myData.src_vocab), params["num_hiddens"], params["ffn_num_hiddens"], params["num_heads"], params["num_blks"], params["dropout"])
    decoder = model.TransformerDecoder(len(myData.tgt_vocab), params["num_hiddens"], params["ffn_num_hiddens"], params["num_heads"], params["num_blks"], params["dropout"])
    myModel = model.Seq2Seq(encoder, decoder, tgt_pad=myData.tgt_vocab['<pad>'])

    myTrainer = trainer.trainer(myData)

    return myModel, myData, myTrainer

device = settings.device
torch.set_default_device(device)
batchSize = 16 # for higher tensor max_len values, keep batchSize smaller
myModel, myData, myTrainer = load_model_data_trainer(modelName)

myTrainer.fit(myModel, 0.0001, epochs=1, showTranslations=False, loadModel=False, shutDown=False, modelName = modelName, calcBleu=True, bleuPriority=True, fromBest = True)

# myModel.loadDict(modelName)
# myModel.load_state_dict(torch.load(f'{os.getcwd()}//models//{modelName}'))
# myTrainer.eval_cycle(myModel, modelName, showTranslations=False, calcBleu=True)

# decode.greedy_eval(100, myModel, myData, myTrainer)
# decode.beam_eval(1, myModel, myData, myTrainer)
# decode.decoder_eval(myModel, myTrainer, 10)

# compare_models(modelDict.keys())
# compare_models(["Full", "Small"])
# show_results()

'''
ENCODER
words to trainable word embeddings
add positional encoding
linear transformations applied to q, k, and v
copied to n heads
embeddings used as q, k, and v, performing dot product self attention
dot product q and k, normalize, dot output and v
concatenate output from multiple heads, pass through linear to get desired length
pass through feed forward network (2 dense layers)

DECODER PREDICTION
bos to word embedding as state
self attention with state and encoder input
take argmax as prediction and append to state
repeat process with new state until max length
'''