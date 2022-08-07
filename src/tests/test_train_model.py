import unittest
import datasets

import jax
import chex
from flax.jax_utils import replicate, unreplicate
import optax
from transformers import FlaxViTForImageClassification

from ..config import DataConfig, ModelConfig, PipelineConfig
from ..data.make_dataset import apply_feature_extraction, create_raw_dataset
from ..models.train_model import (
    get_train_dataloader,
    _train_step,
    ModelParams,
    ShardedModelParams,
)

Dataset = datasets.arrow_dataset.Dataset
test_image_path = "data/testing/raw/"
test_model_name = "google/vit-base-patch16-224"


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.data_args = DataConfig(raw_image_path=test_image_path)
        self.model_args = ModelConfig(model_name=test_model_name)
        self.pipeline_args = PipelineConfig(train_per_device_batch_size=2)

        raw_dataset = create_raw_dataset(self.data_args)
        self.processed_dataset = apply_feature_extraction(
            raw_dataset, self.model_args, self.pipeline_args
        )

    def test_dataloader_output(self):
        train_dataset = self.processed_dataset["train"]
        train_dataloader = get_train_dataloader(train_dataset, self.pipeline_args)

        actual_batch_size = (
            self.pipeline_args.train_per_device_batch_size * jax.device_count()
        )
        num_batches = len(train_dataset) // actual_batch_size
        num_batches_yielded = 0

        for sharded_batch in train_dataloader:
            chex.assert_axis_dimension(
                sharded_batch.pixel_values,
                1,
                self.pipeline_args.train_per_device_batch_size,
            )
            chex.assert_axis_dimension(
                sharded_batch.pixel_values, 0, jax.device_count()
            )
            num_batches_yielded += 1

        self.assertEqual(num_batches_yielded, num_batches)


class StepTrainingLoop(unittest.TestCase):
    """
    Apply one step of the training loop to
    ensure that loss was indeed reduced.
    """

    def setUp(self):
        self.data_args = DataConfig(raw_image_path=test_image_path)
        self.model_args = ModelConfig(model_name=test_model_name)
        self.pipeline_args = PipelineConfig(train_per_device_batch_size=2)

        raw_dataset = create_raw_dataset(self.data_args)
        self.processed_dataset = apply_feature_extraction(
            raw_dataset, self.model_args, self.pipeline_args
        )

    def test_train_step_loss_reduction(self):
        train_dataset = self.processed_dataset["train"]
        train_dataloader = get_train_dataloader(train_dataset, self.pipeline_args)

        model = FlaxViTForImageClassification.from_pretrained(
            self.model_args.model_name
        )  # type: ignore
        model: FlaxViTForImageClassification
        model_params: ModelParams = model.params
        replicated_model_params: ShardedModelParams = replicate(model_params)

        # Important: optimizer should be initialized on non-replicated model params.
        optimizer = optax.adamw(self.model_args.learning_rate)
        optimizer_state = replicate(optimizer.init(model_params))

        batch = next(train_dataloader)
        train_step_output = _train_step(
            batch, model, replicated_model_params, optimizer, optimizer_state
        )
        next_train_step_output = _train_step(
            batch,
            model,
            train_step_output.updated_model_params,
            optimizer,
            train_step_output.updated_optimizer_state,
        )

        self.assertLess(
            unreplicate(next_train_step_output.train_loss),
            unreplicate(train_step_output.train_loss),
        )
