import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 32 # Maximum tokens per sentence
MIN_LEN = 4 # Minimum tokens per sentence
MODEL_NAME = "Small" # Determines model parameters from modelDict in settings.py
MAX_TOKENS = 3000 # Set lower for larger models or less processing power

modelDict = {
    "Full": {"num_hiddens": 512, "num_blks": 6, "dropout": 0.1, "ffn_num_hiddens": 2048, "num_heads": 8},
    "FullNoDrop": {"num_hiddens": 512, "num_blks": 6, "dropout": 0.0, "ffn_num_hiddens": 2048, "num_heads": 8},
    "FullHalfDrop": {"num_hiddens": 512, "num_blks": 6, "dropout": 0.05, "ffn_num_hiddens": 2048, "num_heads": 8},
    "Full3": {"num_hiddens": 512, "num_blks": 3, "dropout": 0.1, "ffn_num_hiddens": 2048, "num_heads": 4},
    "Full3NoDrop": {"num_hiddens": 512, "num_blks": 3, "dropout": 0, "ffn_num_hiddens": 2048, "num_heads": 4},
    "Half": {"num_hiddens": 256, "num_blks": 3, "dropout": 0.1, "ffn_num_hiddens": 1024, "num_heads": 4},
    "Small": {"num_hiddens": 128, "num_blks": 2, "dropout": 0.1, "ffn_num_hiddens": 512, "num_heads": 2},
    "SmallHDrop": {"num_hiddens": 128, "num_blks": 2, "dropout": 0.05, "ffn_num_hiddens": 512, "num_heads": 2},
    "SmallQDrop": {"num_hiddens": 128, "num_blks": 2, "dropout": 0.025, "ffn_num_hiddens": 512, "num_heads": 2},
    "SmallTinyDrop": {"num_hiddens": 128, "num_blks": 2, "dropout": 0.001, "ffn_num_hiddens": 512, "num_heads": 2},
    "SmallNoDrop": {"num_hiddens": 128, "num_blks": 2, "dropout": 0, "ffn_num_hiddens": 512, "num_heads": 2},
    }