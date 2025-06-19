
def check_if_sequence_model(model_cfg):
    return model_cfg.model.type in ["gru", "lstm"]
