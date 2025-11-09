import torch, trainer, model, data, settings, decode, os, logging
from settings import modelDict, MODEL_PARAMS, MAX_TOKENS, SRC_LANG, TRG_LANG

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w')
logger = logging.getLogger(__name__)


def load_model_data_trainer(modelParams):
    params = modelDict[modelParams]
    myData = data.europarl_data(MAX_TOKENS)

    encoder = model.TransformerEncoder(myData.tokenizer.get_vocab_size(), params["num_hiddens"], params["ffn_num_hiddens"], params["num_heads"], params["num_blks"], params["dropout"])
    decoder = model.TransformerDecoder(myData.tokenizer.get_vocab_size(), params["num_hiddens"], params["ffn_num_hiddens"], params["num_heads"], params["num_blks"], params["dropout"])
    myModel = model.Seq2Seq(encoder, decoder, tgt_pad=myData.tokenizer.token_to_id('<pad>'))

    myTrainer = trainer.trainer(myData)

    return myModel, myData, myTrainer

device = settings.device
torch.set_default_device(device)
modelName = MODEL_PARAMS + "_" + SRC_LANG + "-" + TRG_LANG
myModel, myData, myTrainer = load_model_data_trainer(MODEL_PARAMS)

try:
    myTrainer.fit(myModel, epochs=4, showTranslations=False, loadModel=False, shutDown=settings.SHUTDOWN_ON_COMPLETE, modelName = modelName, calcBleu=True, bleuPriority=False, fromBest = True)
except Exception as e:
    logger.exception("Exception occurred", exc_info=e)
    if settings.SHUTDOWN_ON_ERROR:
        os.system('shutdown -s')
# myModel.loadDict(modelName)
# myTrainer.eval_cycle(myModel, showTranslations=False, calcBleu=True)

# decode.greedy_eval(100, myModel, myData)
# decode.beam_eval(myModel, 100)
# decode.compare_vs_forward(myModel, 10)

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