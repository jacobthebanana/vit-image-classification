"""
Functions and tools for training the model. 
Includes train dataloader.
"""

from typing import Tuple, Dict, Callable, Iterator, Any, NamedTuple
from typing_extensions import Literal

import jax
import jax.numpy as jnp
import optax
from flax.training.common_utils import shard
from transformers import FlaxViTForImageClassification
from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput
from datasets.arrow_dataset import Dataset

from ..config import PipelineConfig

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


TrainStepOutput = SingleShardTrainStepOutput


def _loss_fn_single_shard(
    batch: Batch, model: FlaxViTForImageClassification, model_params: Dict
) -> float:
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
    float
        Loss of the given model over this batch.
    """
    pixel_values = batch.pixel_values
    labels = batch.label

    output: FlaxSequenceClassifierOutput = model(pixel_values, model_params, train=True)  # type: ignore
    prediction_logits: Array = output.logits
    batch_losses = optax.softmax_cross_entropy_with_integer_labels(
        prediction_logits, labels
    )

    return jnp.mean(batch_losses)


_value_grad_fn_single_shard: Callable[
    [Batch, FlaxViTForImageClassification, ModelParams],
    Tuple[Array, ModelParamGradients],
] = jax.value_and_grad(_loss_fn_single_shard, argnums=2)


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
    train_loss, model_param_gradients = _value_grad_fn_single_shard(
        batch, model, model_params
    )
    train_loss: Array = jax.lax.pmean(train_loss, axis_name="data")
    avg_model_param_gradients = jax.lax.pmean(model_param_gradients, axis_name="data")

    param_updates, updated_optimizer_state = optimizer.update(
        avg_model_param_gradients, optimizer_state, model_params
    )
    updated_model_params = optax.apply_updates(model_params, param_updates)

    return SingleShardTrainStepOutput(
        updated_model_params=updated_model_params,
        updated_optimizer_state=updated_optimizer_state,
        train_loss=train_loss,
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


def get_train_dataloader(
    dataset: Dataset, pipeline_args: PipelineConfig
) -> Iterator[ShardedBatch]:
    """
    Generate JAX-shardable batch from the dataset.

    Parameters
    ----------
    dataset: Dataset
        Source dataset.

    pipeline_args: PipelineConfig
        Specifies seed and training batch size.
    """
    dataset_length = len(dataset)
    actual_batch_length = pipeline_args.train_per_device_batch_size * jax.device_count()
    num_batches = dataset_length // actual_batch_length

    dataset_indices = jnp.arange(num_batches * actual_batch_length)

    dataloader_prng_key: jax.random.KeyArray = jax.random.PRNGKey(
        pipeline_args.train_dataloader_prng_key
    )
    dataset_indices_shuffled = jax.random.shuffle(dataloader_prng_key, dataset_indices)
    dataset_indices_shuffled = dataset_indices_shuffled.reshape(
        (num_batches, actual_batch_length)
    )

    for batch_index in range(num_batches):
        batch_dataset_indices = dataset_indices_shuffled[batch_index, :]
        raw_batch: RawBatch = dataset[batch_dataset_indices]

        batch: Batch = Batch(
            pixel_values=jnp.array(raw_batch["pixel_values"]),
            label=jnp.array(raw_batch["label"]),
        )
        sharded_batch: ShardedBatch = shard(batch)
        yield sharded_batch
