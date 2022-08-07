from typing import Dict, Any
import json
from collections import Counter

import datasets
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import ViTFeatureExtractor, BatchFeature
from transformers.hf_argparser import HfArgumentParser
from PIL import ImageFile

from ..config import DataConfig, ModelConfig, PipelineConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True
Dataset = datasets.arrow_dataset.Dataset


def create_raw_dataset(data_args: DataConfig) -> DatasetDict:
    """
    Create HuggingFace Image dataset from raw_image_path.
    Prepare the folder as in
    https://huggingface.co/docs/datasets/image_load#imagefolder.

    Parameters
    ----------
    data_args : DataConfig
        Specifies path to the raw image folder.

    Returns
    -------
    Dataset
        HuggingFace Dataset of raw images.
    """
    dataset: DatasetDict = load_dataset("imagefolder", data_dir=data_args.raw_image_path)  # type: ignore
    dataset_split: DatasetDict = dataset["train"].train_test_split(
        test_size=data_args.test_ratio,
        train_size=data_args.train_ratio,
        shuffle=True,
        stratify_by_column="label",
        seed=0,
    )
    return dataset_split


def apply_feature_extraction(
    raw_dataset: DatasetDict, model_args: ModelConfig, pipeline_args: PipelineConfig
) -> DatasetDict:
    """
    Apply the ViT Feature Extractor to the raw dataset.

    Parameters
    ----------
    raw_dataset : Dataset
        HuggingFace Dataset of raw images.
    model_args: ModelConfig
        Specifies the ViT model and feature extractor.

    Returns
    -------
    Dataset
        HuggingFace Dataset of image features.
    """
    feature_extractor: ViTFeatureExtractor = ViTFeatureExtractor.from_pretrained(
        model_args.base_model_name
    )

    def apply_feature_extractor(batch: Dict[str, Any]) -> BatchFeature:
        return feature_extractor(batch["image"])

    return raw_dataset.map(
        apply_feature_extractor, batched=True, num_proc=pipeline_args.num_proc
    )


def get_dataset_label_stats(dataset: Dataset) -> Dict[int, int]:
    """
    Return the number of entries under each label.

    Parameters
    ----------
    dataset: Dataset
        A dataset split (e.g., dataset["train"]).

    Returns
    -------
    Dict[int, int]
        Dictionary mapping label to number of items.
    """
    return Counter(dataset[:]["label"])


def main():
    arg_parser = HfArgumentParser((DataConfig, ModelConfig, PipelineConfig))
    data_args, model_args, pipeline_args = arg_parser.parse_args_into_dataclasses()
    data_args: DataConfig
    model_args: ModelConfig
    pipeline_args: PipelineConfig

    raw_dataset = create_raw_dataset(data_args)
    processed_dataset = apply_feature_extraction(raw_dataset, model_args, pipeline_args)
    processed_dataset.save_to_disk(data_args.processed_dataset_path)

    train_dataset_stats = get_dataset_label_stats(processed_dataset["train"])
    print("Train split", json.dumps(train_dataset_stats, indent=2))

    test_dataset_stats = get_dataset_label_stats(processed_dataset["test"])
    print("Test split", json.dumps(test_dataset_stats, indent=2))


if __name__ == "__main__":
    main()
