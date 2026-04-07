from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_METADATA = {
    "accelerator": "GPU",
    "colab": {"gpuType": "H100", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"{line}\n" for line in source.splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"{line}\n" for line in source.splitlines()],
    }


def build_notebook(
    *,
    title: str,
    task_key: str,
    env_prefix: str,
    output_model_path: str,
    default_workspace: str,
    default_project: str,
    default_version: int,
    primary_goal: str,
) -> dict:
    notebook_source = [
        markdown_cell(
            f"# Experiment: {title}\n\n"
            f"Objective:\n"
            f"- Train a stronger {task_key} detector for the local CourtVision pipeline.\n"
            f"- Default to the YOLO26 family because it is the latest Ultralytics detector family supported by Roboflow and Ultralytics.\n"
            f"- Publish the final checkpoint to `{output_model_path}` so the app can load it directly."
        ),
        markdown_cell(
            "## Model choice\n\n"
            "This notebook is optimized for the current repo, not just for standalone benchmark scores.\n"
            "- Default profile: `balanced`.\n"
            "- Recommended family: `YOLO26`.\n"
            "- Why not switch to Roboflow-hosted RF-DETR here: the backend currently consumes local Ultralytics `.pt` weights directly, so YOLO26 is the strongest drop-in upgrade.\n"
            "- Override with env vars if you want a heavier `quality` run or a lighter `edge` run."
        ),
        code_cell(
            "# Install dependencies once per fresh notebook runtime\n"
            "%pip install -q \"roboflow>=1.2.13\" \"ultralytics>=8.4.14\" pyyaml pillow"
        ),
        code_cell(
            "from __future__ import annotations\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "from ultralytics import YOLO\n\n"
            "repo_root = Path.cwd().resolve()\n"
            "if repo_root.name == \"Training_Notebooks\":\n"
            "    repo_root = repo_root.parent\n"
            "if not (repo_root / \"Training_Notebooks\").exists():\n"
            "    raise FileNotFoundError(\"Run this notebook from the repository root or the Training_Notebooks directory.\")\n\n"
            "training_dir = repo_root / \"Training_Notebooks\"\n"
            "if str(training_dir) not in sys.path:\n"
            "    sys.path.insert(0, str(training_dir))\n\n"
            "from yolo_training_common import (\n"
            "    build_dataset_summary,\n"
            "    build_training_config,\n"
            "    build_training_kwargs,\n"
            "    copy_best_weights,\n"
            "    download_dataset,\n"
            "    inspect_dataset_images,\n"
            "    normalize_data_yaml,\n"
            "    write_training_summary,\n"
            ")\n\n"
            "config = build_training_config(\n"
            f"    task_key=\"{task_key}\",\n"
            f"    env_prefix=\"{env_prefix}\",\n"
            f"    output_model_path=\"{output_model_path}\",\n"
            f"    default_workspace=\"{default_workspace}\",\n"
            f"    default_project=\"{default_project}\",\n"
            f"    default_version={default_version},\n"
            ")\n"
            "config"
        ),
        markdown_cell(
            "## Plan\n\n"
            "- Download the chosen Roboflow dataset version.\n"
            "- Normalize `data.yaml` so the notebook does not depend on folder-moving hacks.\n"
            "- Inspect class labels and sample image resolution before training.\n"
            f"- Train the detector with a preset tuned for {primary_goal}.\n"
            "- Validate the best checkpoint and copy it into `Models/`.\n"
            "- Write a small training metadata JSON file next to the exported model."
        ),
        code_cell(
            "dataset_root = download_dataset(config)\n"
            "data_yaml_path = normalize_data_yaml(dataset_root)\n"
            "image_stats = inspect_dataset_images(data_yaml_path)\n"
            "dataset_summary = build_dataset_summary(config, data_yaml_path, image_stats)\n"
            "dataset_summary"
        ),
        code_cell(
            "model = YOLO(config[\"base_model\"])\n"
            "train_results = model.train(**build_training_kwargs(config, data_yaml_path))\n\n"
            "save_dir = Path(train_results.save_dir)\n"
            "best_weights_path = save_dir / \"weights\" / \"best.pt\"\n"
            "save_dir"
        ),
        code_cell(
            "trained_model = YOLO(best_weights_path)\n"
            "val_metrics = trained_model.val(\n"
            "    data=str(data_yaml_path),\n"
            "    imgsz=config[\"image_size\"],\n"
            "    batch=config[\"batch_size\"],\n"
            "    split=\"val\",\n"
            "    device=config[\"device\"],\n"
            ")\n\n"
            "published_model_path = copy_best_weights(save_dir, config[\"output_model_path\"])\n"
            "summary_path = write_training_summary(\n"
            "    config=config,\n"
            "    dataset_summary=dataset_summary,\n"
            "    best_weights_path=best_weights_path,\n"
            "    published_model_path=published_model_path,\n"
            "    map50=float(val_metrics.box.map50),\n"
            "    map50_95=float(val_metrics.box.map),\n"
            ")\n\n"
            "result = {\n"
            "    \"task_name\": config[\"task_name\"],\n"
            "    \"profile\": config[\"profile\"],\n"
            "    \"base_model\": config[\"base_model\"],\n"
            "    \"best_weights_path\": str(best_weights_path),\n"
            "    \"published_model_path\": str(published_model_path),\n"
            "    \"summary_path\": str(summary_path),\n"
            "    \"map50\": float(val_metrics.box.map50),\n"
            "    \"map50_95\": float(val_metrics.box.map),\n"
            "}\n"
            "result"
        ),
        markdown_cell(
            "## Next steps\n\n"
            "- If the dataset warnings mention low export resolution, regenerate the Roboflow dataset version at a larger size before retraining.\n"
            f"- If this run is too slow for the app, switch to `COURTVISION_{env_prefix}_PROFILE=edge`.\n"
            f"- If recall is still weak and local speed is acceptable, switch to `COURTVISION_{env_prefix}_PROFILE=quality`.\n"
            "- Keep the generated `*.train.json` file with the exported model so you know which dataset version produced it."
        ),
    ]

    return {
        "cells": notebook_source,
        "metadata": NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebooks = {
        repo_root / "Training_Notebooks" / "basketball_ball_detection_training.ipynb": build_notebook(
            title="Basketball Ball Detection Training",
            task_key="ball",
            env_prefix="BALL",
            output_model_path="Models/ball_detector_model.pt",
            default_workspace="cricket-qnb5l",
            default_project="basketball-xil7x",
            default_version=1,
            primary_goal="small-object recall without making the final detector unnecessarily heavy",
        ),
        repo_root / "Training_Notebooks" / "basketball_player_detection_training.ipynb": build_notebook(
            title="Basketball Player Detection Training",
            task_key="player",
            env_prefix="PLAYER",
            output_model_path="Models/Player_detection_model.pt",
            default_workspace="cricket-qnb5l",
            default_project="basketball-xil7x",
            default_version=1,
            primary_goal="crowded player detection while keeping inference practical on a MacBook Air",
        ),
    }

    for path, notebook in notebooks.items():
        path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
