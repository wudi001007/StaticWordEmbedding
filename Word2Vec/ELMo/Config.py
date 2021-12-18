
Config = {
    "max_tok_len": 50,
    "train_file": "./train.txt",
    "model_path": "./elmo_bilm",
    "char_embedding_dim": 50,
    "char_conv_filter": [[1,32], [2,32]],
    "num_highways": 2,
    "projection_dim": 512,
    "hidden_dim": 4096,
    "num_layers": 2,
    "batch_size": 32,
    "dropout_prob":0.1,
    "learning_rate": 0.0004,
    "clip_grad": 5,
    "num_epoch": 10
}