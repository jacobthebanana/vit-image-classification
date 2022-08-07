# Vision Transformer Image Classification with Transfer Learning
Scripts for fine-tuning Vision Transformer (ViT) models.
Proudly built with JAX, Optax, Flax, HuggingFace Transformers, 
HuggingFace Datasets, and wandb.ai.

## Overview
This project uses GNU Make (`makefile`) for reproducibility. 
Python dependencies are listed under requirements.txt.

Thanks to JAX, the same codebase works on GPUs and TPUs with data
parallelism enabled. 

## Setup
### Dependencies
Dependencies of this project are listed under `requirements.txt`.
[Additional steps](https://github.com/google/jax#installation) 
might be required to install the appropriate version of  `jaxlib` on your system. 

### Data 
Arrange your data in the `imagefolder` format, as in the 
[HuggingFace Datasets documentation](https://huggingface.co/docs/datasets/image_load#imagefolder).

### Hyperparameters, Preprocessing, and Training
Hyperparameters are specified in `.env` and are excluded from version control.
Be sure to make a copy of `env.example` as all entries are required:
```bash
cp env.example .env
```

Adjust hyperparameters in `.env`. 

Make `preprocess_and_train` to preprocess the data and train the model.
```bash
make preprocess_and_train
```


## Contributing
### Unit Tests
This project is unit tested on a small subset of the train dataset:
```bash
# Generate subset for unit testing.
make setup_test_data 

# Run unit tests.
make run_tests
```