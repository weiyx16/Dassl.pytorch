"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import json
import warnings
import torch
import torch.nn as nn

from .radam import RAdam

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]


def build_optimizer(model, optim_cfg, assigner=None):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """
    optim = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    layer_decay = optim_cfg.LAYER_DECAY
    base_lr_mult = optim_cfg.BASE_LR_MULT

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            "Unsupported optim: {}. Must be one of {}".format(
                optim, AVAI_OPTIMS
            )
        )

    if staged_lr:
        if not isinstance(model, nn.Module):
            raise TypeError(
                "When staged_lr is True, model given to "
                "build_optimizer() must be an instance of nn.Module"
            )

        if isinstance(model, nn.DataParallel):
            model = model.module

        if isinstance(new_layers, str):
            if new_layers is None:
                warnings.warn(
                    "new_layers is empty, therefore, staged_lr is useless"
                )
            new_layers = [new_layers]

        base_params = []
        base_layers = []
        new_params = []

        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)

        param_groups = [
            {
                "params": base_params,
                "lr": lr * base_lr_mult
            },
            {
                "params": new_params
            },
        ]

    elif layer_decay != 1.0:
        if not isinstance(model, nn.Module):
            raise TypeError(
                "When layer_decay is not 1.0, model given to "
                "build_optimizer() must be an instance of nn.Module"
            )

        if isinstance(model, nn.DataParallel):
            model = model.module

        # the assigner has two function: one: name to layer_id; two: name to layer decay value
        get_num_layer = assigner.get_layer_id
        get_layer_scale=assigner.get_scale

        parameter_group_names = {}
        parameter_group_vars = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
                
            layer_id = get_num_layer(name)
            group_name = "layer_decay_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_group_names:
                scale = get_layer_scale(layer_id)
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale,
                    # "lr": scale * lr
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale,
                    # "lr": scale * lr
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        param_groups = list(parameter_group_vars.values())

    else:
        if isinstance(model, nn.Module):
            param_groups = model.parameters()
        else:
            param_groups = model

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "radam":
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    return optimizer
