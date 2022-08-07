import unittest
import datasets

from ..config import DataConfig, ModelConfig, PipelineConfig
from ..data.make_dataset import create_raw_dataset, apply_feature_extraction

Dataset = datasets.arrow_dataset.Dataset
test_image_path = "data/testing/raw/"
test_model_name = "google/vit-base-patch16-224"


class CreateDatasetFromImageFolder(unittest.TestCase):
    def setUp(self):
        self.data_args = DataConfig(raw_image_path=test_image_path)
        self.raw_dataset = create_raw_dataset(self.data_args)

    def test_dataset_classes(self):
        train_dataset: Dataset = self.raw_dataset["train"]  # type: ignore
        self.assertEqual(
            len(train_dataset), int((10 + 17) * self.data_args.train_ratio)
        )


class ExtractImageFeatures(unittest.TestCase):
    def setUp(self):
        self.data_args = DataConfig(raw_image_path=test_image_path)
        self.model_args = ModelConfig(model_name=test_model_name)
        self.pipeline_args = PipelineConfig(num_proc=1)
        self.raw_dataset = create_raw_dataset(self.data_args)

    def test_apply_feature_extraction(self):
        processed_dataset = apply_feature_extraction(
            self.raw_dataset, self.model_args, self.pipeline_args
        )
        train_dataset: Dataset = self.raw_dataset["train"]  # type: ignore
        processed_train_dataset: Dataset = processed_dataset["train"]  # type: ignore

        self.assertEqual(
            len(processed_train_dataset),
            len(train_dataset),
        )
        self.assertEqual(len(processed_train_dataset["pixel_values"][0]), 3)
        self.assertEqual(len(processed_train_dataset["pixel_values"][0][0]), 224)
        print(processed_dataset)
