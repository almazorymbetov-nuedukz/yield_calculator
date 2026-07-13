import torch

from yield_calc.data import build_molecular_cluster_dataset
from yield_calc.modules import TransferLearningYieldNet


def test_build_molecular_cluster_dataset_returns_numeric_features():
    cluster = {
        "name": "des_cluster",
        "components": [
            {"name": "choline", "role": "hba", "count": 1},
            {"name": "glycerol", "role": "hbd", "count": 2},
            {"name": "methyl_oleate", "role": "biodiesel", "count": 1},
        ],
    }

    payload = build_molecular_cluster_dataset([cluster], output_path=None)

    assert payload["clusters"][0]["name"] == "des_cluster"
    assert payload["feature_names"]
    assert payload["feature_matrix"].shape[0] == 1
    assert payload["feature_matrix"].shape[1] == len(payload["feature_names"])
    assert payload["feature_matrix"][0, 0] >= 0


def test_transfer_learning_model_forward_shape():
    model = TransferLearningYieldNet(input_dim=8, hidden_dim=16, pretrained_dim=8)
    x = torch.randn(3, 8)
    out = model(x)

    assert out.shape == (3, 1)
