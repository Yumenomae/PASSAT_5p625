from .update import update

def build_update_method(config):
    if config.MODEL.TYPE == 'PASSAT':
        method = update
    else:
        raise NotImplementedError(f"Unkown method: {config.MODEL.TYPE}")
    return method
