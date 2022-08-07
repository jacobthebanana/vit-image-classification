"""
Functions and tools for training the model. 
Includes train dataloader.
"""

from typing import Tuple, Dict, Callable, Iterator, Any, NamedTuple, Optional
from typing_extensions import Literal

import jax
import jax.numpy as jnp
import optax
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
from transformers import FlaxViTForImageClassification
from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput
from transformers.hf_argparser import HfArgumentParser
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from tqdm.auto import tqdm
from PIL import ImageFile

import datetime
from socket import gethostname
import wandb

from ..config import PipelineConfig, ModelConfig, DataConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True
Array = jnp.ndarray

# Batch sampled from the dataset.
RawBatchKeys = Literal["image", "label", "pixel_values"]
RawBatch = Dict[RawBatchKeys, Any]

# Batch as a JAX tree that could be shard across accelerators.
class Batch(NamedTuple):
    pixel_values: Array
    label: Array


ShardedBatch = Batch
ModelParams = optax.Params
ShardedModelParams = ModelParams
ModelParamGradients = ModelParams


class SingleShardTrainStepOutput(NamedTuple):
    updated_model_params: ModelParams
    updated_optimizer_state: optax.OptState
    train_loss: Array
    train_accuracy: Array


class EvalStepOutput(NamedTuple):
    loss: float
    accuracy: float


TrainStepOutput = SingleShardTrainStepOutput


def _loss_fn_single_shard(
    batch: Batch, model: FlaxViTForImageClassification, model_params: ModelParams
) -> Tuple[float, float]:
    """
    Loss function for a batch that was not sharded.

    Parameters
    ----------
    batch : Batch
        batch of examples that were not sharded.

    model : FlaxViTForImageClassification
        abstract Flax model.

    model_params : Dict
        Flax model params, not sharded.

    Returns
    -------
    float, float
        Loss and accuracy of the given model over this batch.
    """
    pixel_values = batch.pixel_values
    labels = batch.label

    output: FlaxSequenceClassifierOutput = model(pixel_values, model_params, train=True)  # type: ignore
    prediction_logits: Array = output.logits
    predictions: Array = jnp.argmax(prediction_logits, axis=1)
    num_correct = jnp.sum(predictions == labels)
    fraction_correct = num_correct / predictions.shape[0]

    batch_losses = optax.softmax_cross_entropy_with_integer_labels(
        prediction_logits, labels
    )

    return jnp.mean(batch_losses), fraction_correct


_loss_fn: Callable[
    [ShardedBatch, FlaxViTForImageClassification, ShardedModelParams], Array
] = jax.pmap(_loss_fn_single_shard, static_broadcasted_argnums=(1,))


_value_grad_fn_single_shard: Callable[
    [Batch, FlaxViTForImageClassification, ModelParams],
    Tuple[Tuple[Array, Array], ModelParamGradients],
] = jax.value_and_grad(_loss_fn_single_shard, argnums=2, has_aux=True)


def _train_step_single_shard(
    batch: Batch,
    model: FlaxViTForImageClassification,
    model_params: ModelParams,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
) -> SingleShardTrainStepOutput:
    """
    Calculate loss and updated model_params on the given batch.

    Parameters
    ----------
    batch: Batch
        Non-sharded.

    model: FlaxViTForImageClassification
        Abstract Flax model.

    model_params: ModelParams
        Model parameters.

    Returns
    ---
    SingleShardTrainStepOutput
        Note that the model params are processed with jax.pmean
        over the data-parallelism axis. Additionally, the
        optimizer state isn't averaged across accelerators.

    """
    (train_loss, train_accuracy), model_param_gradients = _value_grad_fn_single_shard(
        batch, model, model_params
    )
    train_loss: Array = jax.lax.pmean(train_loss, axis_name="data")
    train_accuracy: Array = jax.lax.pmean(train_accuracy, axis_name="data")
    avg_model_param_gradients = jax.lax.pmean(model_param_gradients, axis_name="data")

    param_updates, updated_optimizer_state = optimizer.update(
        avg_model_param_gradients, optimizer_state, model_params
    )
    updated_model_params = optax.apply_updates(model_params, param_updates)

    return SingleShardTrainStepOutput(
        updated_model_params=updated_model_params,
        updated_optimizer_state=updated_optimizer_state,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
    )


_train_step: Callable[
    [
        Batch,
        FlaxViTForImageClassification,
        ShardedModelParams,
        optax.GradientTransformation,
        optax.OptState,
    ],
    SingleShardTrainStepOutput,
] = jax.pmap(
    _train_step_single_shard, axis_name="data", static_broadcasted_argnums=(1, 3)
)


def get_dataloader(
    dataset: Dataset,
    per_device_batch_size: int,
    prng_key: int = 0,
    shuffle: bool = True,
) -> Iterator[ShardedBatch]:
    """
    Generate JAX-shardable batch from the dataset.

    Parameters
    ----------
    dataset: Dataset
        Source dataset.

    per_device_batch_size: int
        Batch size per accelerator device, such that the actual
        batch size would be per_device_batch_size * jax.device_count()

    shuffle: bool
        Specifies whether to shuffle the dataset.

    prng_key: int
        Only if shuffling is enabled.

    Yields
    ------
    Iterator[ShardedBatch]
        Iterator of sharded data batches
    """
    dataset_length = len(dataset)
    actual_batch_length = per_device_batch_size * jax.device_count()
    num_batches = dataset_length // actual_batch_length

    dataset_indices = jnp.arange(num_batches * actual_batch_length)

    if shuffle:
        dataloader_prng_key: jax.random.KeyArray = jax.random.PRNGKey(prng_key)
        dataset_indices = jax.random.permutation(
            dataloader_prng_key, dataset_indices, independent=True
        )

    dataset_indices = dataset_indices.reshape((num_batches, actual_batch_length))

    for batch_index in range(num_batches):
        batch_dataset_indices = dataset_indices[batch_index, :]
        raw_batch: RawBatch = dataset[batch_dataset_indices]

        batch: Batch = Batch(
            pixel_values=jnp.array(raw_batch["pixel_values"]),
            label=jnp.array(raw_batch["label"]),
        )
        sharded_batch: ShardedBatch = shard(batch)
        yield sharded_batch


def get_test_loss(
    test_dataloader: Iterator[ShardedBatch],
    model: FlaxViTForImageClassification,
    model_params: ShardedModelParams,
    num_batches: int,
) -> EvalStepOutput:
    """
    Given a dataloader, return the average
    loss of the model on the examples.

    Parameters
    ----------
    test_dataloader: Iterator[ShardedBatch]
        yields sharded examples.

    model: FlaxViTForImageClassification
        abstract Flax model.

    model_params: ModelParams
        parameters for the Flax model.


    Returns
    -------
    EvalStepOutput:
        Avg. loss and avg. accuracy across the examples.
    """
    loss_tally = 0
    accuracy_tally = 0

    for batch in tqdm(
        test_dataloader, total=num_batches, desc="Testing", ncols=80, leave=False
    ):
        batch_loss, batch_accuracy = unreplicate(_loss_fn(batch, model, model_params))

        loss_tally += batch_loss
        accuracy_tally += batch_accuracy

    return EvalStepOutput(
        loss=loss_tally / num_batches, accuracy=accuracy_tally / num_batches
    )


def main():
    """
    Args:
     data_args.processed_dataset_path
     pipeline_args.train_per_device_batch_size
     pipeline_args.train_dataloader_prng_key
     model_args.base_model_name
     model_args.learning_rate
     model_args.weight_decay
    """

    arg_parser = HfArgumentParser((DataConfig, ModelConfig, PipelineConfig))
    data_args, model_args, pipeline_args = arg_parser.parse_args_into_dataclasses()
    data_args: DataConfig
    model_args: ModelConfig
    pipeline_args: PipelineConfig

    dataset = load_from_disk(data_args.processed_dataset_path)
    print(dataset)

    train_dataset: Dataset = dataset["train"]  # type: ignore
    train_dataset_length = len(train_dataset)
    actual_train_batch_length = (
        pipeline_args.train_per_device_batch_size * jax.device_count()
    )
    num_train_batches = train_dataset_length // actual_train_batch_length
    num_train_steps = pipeline_args.num_epochs * num_train_batches

    test_dataset: Dataset = dataset["test"]  # type: ignore
    test_dataset_length = len(test_dataset)
    actual_test_batch_length = (
        pipeline_args.test_per_device_batch_size * jax.device_count()
    )
    num_test_batches = test_dataset_length // actual_test_batch_length

    model = FlaxViTForImageClassification.from_pretrained(
        model_args.base_model_name
    )  # type: ignore
    model: FlaxViTForImageClassification
    model_params: ModelParams = model.params  # type: ignore
    lr_schedule = optax.linear_schedule(
        init_value=model_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps,
    )
    optimizer = optax.adamw(
        learning_rate=lr_schedule, weight_decay=model_args.weight_decay
    )
    optimizer_state: optax.OptState = optimizer.init(model_params)

    sharded_model_params: ShardedModelParams = replicate(model_params)
    sharded_optimizer_state: optax.OptState = replicate(optimizer_state)

    wandb.init(
        project="zadane-photomath",
        entity="jacobthebanana",
        name=datetime.datetime.now().isoformat() + "-" + gethostname(),
    )
    wandb.run.log_code(".")  # type: ignore
    wandb.config.update(
        {
            "model_args": model_args.__dict__,
            "data_args": data_args.__dict__,
            "pipeline_args": pipeline_args.__dict__,
        }
    )

    for batch_index in tqdm(range(pipeline_args.num_epochs), ncols=80):
        train_dataloader = get_dataloader(
            train_dataset,
            per_device_batch_size=pipeline_args.train_per_device_batch_size,
            prng_key=pipeline_args.train_dataloader_prng_key,
        )

        for batch_index, batch in enumerate(
            tqdm(train_dataloader, total=num_train_batches, leave=False, ncols=80)
        ):
            train_step_output = _train_step(
                batch, model, sharded_model_params, optimizer, sharded_optimizer_state
            )

            sharded_model_params = train_step_output.updated_model_params
            sharded_optimizer_state = train_step_output.updated_optimizer_state
            train_loss: float = unreplicate(train_step_output.train_loss)
            train_accuracy: float = unreplicate(train_step_output.train_accuracy)

            if batch_index % pipeline_args.test_every_num_steps == 0:
                test_dataloader = get_dataloader(
                    test_dataset,
                    per_device_batch_size=pipeline_args.test_per_device_batch_size,
                    shuffle=False,
                )
                eval_output = get_test_loss(
                    test_dataloader,
                    model,
                    sharded_model_params,
                    num_batches=num_test_batches,
                )
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "test_loss": eval_output.loss,
                        "test_accuracy": eval_output.accuracy,
                    }
                )
            else:
                wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})

        model.save_pretrained(
            model_args.model_output_path_prefix, unreplicate(sharded_model_params)
        )


if __name__ == "__main__":
    main()
