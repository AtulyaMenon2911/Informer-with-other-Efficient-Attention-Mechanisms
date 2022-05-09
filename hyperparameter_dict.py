hyperarameter_dict = {
    "ETTh1": {
        "all": {
            "n_heads": [4, 8],  # 2,16
            "weight_decay": [0.1, 0.01, 0.001],
            # "dropout": [0.05, 0.1, 0.15],
            # "e_layers": [2, 3, 4],
            # "d_layers": [1, 2],
            # "seq_len": [24, 48, 96, 168, 336, 480, 720],
            # "label_len": [24, 48, 96, 168, 336, 480, 720],
            # "pred_len": [24, 48, 96, 168, 336, 720],
            # "features": ["S", "M"]
        },
        "qs": {
            "fraction": [0.2, 0.3, 0.4]

        }
    },
    "ETTm1": {
        "all": {
            "n_heads": [4, 8],  # 2,16
            "weight_decay": [0.1, 0.01, 0.001],
            # "dropout": [0.05, 0.1, 0.15],
            # "e_layers": [2, 3, 4],
            # "d_layers": [1, 2],
            # "seq_len": [24, 48, 96, 168, 336, 480, 720],
            # "label_len": [24, 48, 96, 168, 336, 480, 720],
            # "pred_len": [24, 48, 96, 168, 336, 720],
            # "features": ["S", "M"]
        },
        "qs": {
            "fraction": [0.2, 0.3, 0.4]

        }
    },
    "WTH": {
        "all": {
            "n_heads": [4, 8],  # 2,16
            "weight_decay": [0.1, 0.01, 0.001],
            # "dropout": [0.05, 0.1, 0.15],
            # "e_layers": [2, 3, 4],
            # "d_layers": [1, 2],
            # "seq_len": [24, 48, 96, 168, 336, 480, 720],
            # "label_len": [24, 48, 96, 168, 336, 480, 720],
            # "pred_len": [24, 48, 96, 168, 336, 720],
            # "features": ["S", "M"]
        },
        "qs": {
            "fraction": [0.2, 0.3, 0.4]

        }
    },

}


def create_hyperparameter_dict(name_dataset: str, model: str):
    dataset_dict = hyperarameter_dict[name_dataset]["all"]
    model_dict = {}
    if model in hyperarameter_dict[name_dataset]:
        model_dict = hyperarameter_dict[name_dataset][model]
    final_dict = {**dataset_dict, **model_dict}
    return final_dict
