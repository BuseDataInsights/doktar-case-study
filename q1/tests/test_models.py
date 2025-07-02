import pathlib, yaml
from vmc_model import train


def test_end_to_end(tmp_path: pathlib.Path):
    cfg = yaml.safe_load(pathlib.Path("config.yaml").read_text())
    # point to a temp copy of the data to ensure read‑write permissions
    data_copy = tmp_path / "Doktar_Topology_Sample_Data_20250627.xlsx"
    pathlib.Path(cfg["data_file"]).rename(data_copy)
    cfg["data_file"] = str(data_copy)

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump(cfg))

    train.train_best(cfg_path=cfg_file, output=tmp_path / "model.joblib")
    assert (tmp_path / "model.joblib").exists()
