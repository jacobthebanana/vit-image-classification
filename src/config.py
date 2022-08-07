from dataclasses import dataclass, field
import multiprocessing


@dataclass
class DataConfig:
    raw_image_path: str = field(default="data/raw")
    interim_dataset_path: str = field(default="data/interim/")
    processed_dataset_path: str = field(default="data/processed/")
    test_ratio: float = field(default=0.2)


@dataclass
class ModelConfig:
    base_model_name: str = field(default="google/vit-base-patch16-224")
    model_output_path_prefix: str = field(default="data/artifacts/model")
    weight_decay: float = field(default=0.0001)
    learning_rate: float = field(default=0.001)


@dataclass
class PipelineConfig:
    num_proc: int = field(default=multiprocessing.cpu_count())
    num_epochs: int = field(default=6)
    train_dataloader_prng_key: int = field(default=0)
    train_per_device_batch_size: int = field(default=16)
    test_per_device_batch_size: int = field(default=32)
    test_every_num_steps: int = field(default=1)
    wandb_entity: str = field(default="")
    wandb_project: str = field(default="")