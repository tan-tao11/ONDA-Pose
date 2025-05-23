import torch
import socket
import os
import shutil
import time

def get_child_state_dict(state_dict, key):
    return {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("{}.".format(key))}

def restore_checkpoint(cfg, model, load_name=None, resume=False):
    assert ((load_name is None) == (resume is not False))  # resume can be True/False or epoch numbers
    if resume:
        load_name = "{0}/model.ckpt".format(cfg.output) if resume is True else \
            "{0}/model/{1}.ckpt".format(cfg.output, resume)
    checkpoint = torch.load(load_name, map_location=cfg.device)
    # load individual (possibly partial) children modules
    for name, child in model.graph.named_children():
        child_state_dict = get_child_state_dict(checkpoint["graph"], name)
        if child_state_dict:
            print("restoring {}...".format(name))
            child.load_state_dict(child_state_dict)

    # Load optimizer and scheduler
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"] and key in checkpoint and resume:
            print("restoring {}...".format(key))
            getattr(model, key).load_state_dict(checkpoint[key])

    if resume:
        ep, it = checkpoint["epoch"], checkpoint["iter"]
        if resume is not True:  # not boolean
            assert (resume == ep) or (resume == it)
        print("resuming from epoch {0} (iteration {1})".format(ep, it))
    else:
        ep, it = None, None
    return ep, it

def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open


def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"):  # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device)
    return X

def save_checkpoint(cfg, model, epoch, iter, latest=False, children=None):
    """Save a checkpoint of the model state.

    Args:
        cfg: The configuration object.
        model: The model object.
        epoch: The epoch number.
        iter: The iteration number.
        latest: If True, save the checkpoint as the latest one.
        children: A string or a list of strings of the children modules to save.
            If None, save the whole graph.
    """
    os.makedirs("{0}/model".format(cfg.output), exist_ok=True)
    if children is not None:
        if isinstance(children, str):
            children = [children]
        graph_state_dict = {k: v for k, v in model.graph.state_dict().items() if k.startswith(tuple(children))}
    else:
        graph_state_dict = model.graph.state_dict()
    checkpoint = dict(
        epoch=epoch,
        iter=iter,
        graph=graph_state_dict,
    )
    # Save optimizer and scheduler states
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"]:
            checkpoint.update({key: getattr(model, key).state_dict()})
    # Save the checkpoint
    torch.save(checkpoint, "{0}/model.ckpt".format(cfg.output))
    # Save the checkpoint with the iteration number if not latest
    if not latest:
        shutil.copy("{0}/model.ckpt".format(cfg.output),
                    "{0}/model/{1}.ckpt".format(cfg.output, iter))
        
def get_layer_dims(layers):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1], layers[1:]))

def update_timer(cfg, timer, ep, it_per_ep):
    if not cfg.max_epoch: return
    momentum = 0.99
    timer.elapsed = time.time() - timer.start
    timer.it = timer.it_end - timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean * momentum + timer.it * (1 - momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean * it_per_ep * (cfg.max_epoch - ep)

def visual_img(img, tittle, vis):
    for i in range(img.shape[0]):
        if torch.max(img[i]) > 1.0:
            img_i = img[i]/torch.max(img[i])*255
        else:
            img_i = img[i]
        vis.image(
            img_i,
            opts={'title': tittle+str(i)}, win = tittle+str(i))  
        
def restore_pretrain_partial_checkpoint(cfg, model, load_name=None, resume=False):
    assert ((load_name is None) == (resume is not False))  # resume can be True/False or epoch numbers
    if resume:
        load_name = os.path.join(cfg.output.replace("texture", "pretrain"), "model.ckpt") 
    checkpoint = torch.load(load_name, map_location=cfg.device)

    # load individual (possibly partial) children modules
    for name, child in model.graph.named_children():
        child_state_dict = {".".join(k.split(".")[1:]): v for k, v in checkpoint["graph"].items()
                            if 'mlp_feat' in k and k.startswith("{}.".format(name))}
        # child_state_dict = get_child_state_dict(checkpoint["graph"], name)

        if child_state_dict:
            print("restoring pretrained {} mlp_feat for geometric feature extraction...".format(name))
            model_dict = child.state_dict()
            pretrained_dict = {k: v for k, v in child_state_dict.items() if k in child.state_dict()}
            model_dict.update(pretrained_dict)
            child.load_state_dict(model_dict)
    epoch, iter = None, None
    return epoch, iter