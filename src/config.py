from dataclasses import dataclass, field
import multiprocessing


@dataclass
class DataConfig:
    raw_image_path: str = field(default="data/raw")
    interim_dataset_path: str = field(default="data/interim/")
    processed_dataset_path: str = field(default="data/processed/")
    train_ratio: float = field(default=0.7)
    test_ratio: float = field(default=0.2)


@dataclass
class ModelConfig:
    model_name: str = field(default="google/vit-base-patch16-224")
    learning_rate: float = field(default=0.001)


@dataclass
class PipelineConfig:
    num_proc: int = field(default=multiprocessing.cpu_count())
    train_dataloader_prng_key: int = field(default=0)
    train_per_device_batch_size: int = field(default=128)
