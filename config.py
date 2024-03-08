import torch

config = {
    "batch_size": 64,
    "model": {
        "CNN": {
            "learning_rate": 0.001,
            "num_epochs": 75,
            "L2_regularization": 0.0001,
            "early_stopping_rounds": 10,
            "num_classes": 7,
            "input_size": (224, 224),
            "batch_size": 64,
        },
        "resnet": {
            "model_name": "resnet18",
            "unfreeze_layers": ["layer4", "fc"],
            "NN": {
                "learning_rate": 0.001,
                "num_epochs": 200,
                "L2_regularization": 0.001,
                "early_stopping_rounds": 7
            },
            "SVM": {
                "learning_rate": 0.0001,
                "num_epochs": 200,
                "L2_regularization": 0.001,
                "early_stopping_rounds": 7
            },
            "RF": {
                "num_trees": 100,
                "max_depth": 6,
                "num_epochs": 200,
                "early_stopping_rounds": 30,
                "objective": "multi:softprob",
                "num_classes": 7,
                "eval_metric": "mlogloss",
                "use_label_encoder": False
            }
        },
        "vit": {
            "model_name": "google/vit-base-patch16-224-in21k",
            "unfreeze_layers": ["layer.10", "layer.11", "head"],
            "NN": {
                "learning_rate": 0.001,
                "num_epochs": 200,
                "L2_regularization": 0.0001,
                "early_stopping_rounds": 7
            },
            "SVM": {
                "learning_rate": 0.0001,
                "num_epochs": 200,
                "L2_regularization": 0.0001,
                "early_stopping_rounds": 7
            },
            "RF": {
                "num_trees": 100,
                "max_depth": 6,
                "num_epochs": 200,
                "early_stopping_rounds": 45,
                "objective": "multi:softprob",
                "num_classes": 7,
                "eval_metric": "mlogloss",
                "use_label_encoder": False
            }
        },
        "efficientnet": {
            "model_name": "efficientnet_b0",
            "unfreeze_layers": ["features", "classifier"],
            "NN": {
                "learning_rate": 0.001,
                "num_epochs": 200,
                "L2_regularization": 0.001,
                "early_stopping_rounds": 7
            },
            "SVM": {
                "learning_rate": 0.0001,
                "num_epochs": 200,
                "L2_regularization": 0.001,
                "early_stopping_rounds": 7
            },
            "RF": {
                "num_trees": 100,
                "max_depth": 6,
                "num_epochs": 200,
                "early_stopping_rounds": 30,
                "objective": "multi:softprob",
                "num_classes": 7,
                "eval_metric": "mlogloss",
                "use_label_encoder": False
            }
        }
    },
}
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# config["device"] = torch.device("cpu")