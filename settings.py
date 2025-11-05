import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 2000 # Maximum tokens per sentence, can cause OOM error if too high so set accordingly
MIN_LEN = 1 # Minimum tokens per sentence
MODEL_PARAMS = "Small" # Determines model parameters from modelDict in settings.py
MAX_TOKENS = 2000 # Set lower for larger models or less processing power
SRC_LANG = "en" # Source language
TRG_LANG = "de" # Target language
DATA_MODE = "wmt" # europarl or wmt source data
SHUTDOWN = False # Shutdown PC after training completes

dataFileDict = {
    "europarl": {
        "src": f"europarl-v7.fr-en.clean.{SRC_LANG}",
        "tgt": f"europarl-v7.fr-en.clean.{TRG_LANG}"
    },
    "wmt": {
        "src_train": f"train.{SRC_LANG}",
        "tgt_train": f"train.{TRG_LANG}",
        "src_test": f"test.{SRC_LANG}",
        "tgt_test": f"test.{TRG_LANG}"
    }
}

modelDict = {
    "Full": {"num_hiddens": 512, "num_blks": 6, "dropout": 0.1, "ffn_num_hiddens": 2048, "num_heads": 8},
    "Half": {"num_hiddens": 256, "num_blks": 3, "dropout": 0.1, "ffn_num_hiddens": 1024, "num_heads": 4},
    "Small": {"num_hiddens": 128, "num_blks": 2, "dropout": 0.1, "ffn_num_hiddens": 512, "num_heads": 2},
    "SmallNoDrop": {"num_hiddens": 128, "num_blks": 2, "dropout": 0, "ffn_num_hiddens": 512, "num_heads": 2},
    }