include .env

setup_data_folder:
	mkdir -pv data/raw
	mkdir -pv data/raw/train
	mkdir -pv data/raw/train/pos
	mkdir -pv data/raw/train/neg
	mkdir -pv data/interim
	mkdir -pv data/processed
	mkdir -pv data/artifacts

	mkdir -pv data/testing/raw
	mkdir -pv data/testing/raw/train
	mkdir -pv data/testing/raw/train/pos
	mkdir -pv data/testing/raw/train/neg
	mkdir -pv data/testing/interim
	mkdir -pv data/testing/processed
	mkdir -pv data/testing/artifacts


download_data:
	wget -O data/raw/pos.zip "${drive_export_link_prefix}${drive_pos_file_id}"
	wget -O data/raw/neg.zip "${drive_export_link_prefix}${drive_neg_file_id}"

extract_data:
	unzip data/raw/pos.zip -d data/raw/train/pos
	unzip data/raw/neg.zip -d data/raw/train/neg

prepare_data: setup_data_folder download_data extract_data

preprocess_data:
	python3 -m src.data.make_dataset \
		--raw_image_path="data/raw" \
		--processed_dataset_path="data/processed" \
		--train_ratio=${train_ratio} \
		--train_ratio=${test_ratio} \
		--base_model_name=${base_model_name}

train_model:
	python3 -m src.models.train_model \
		--processed_dataset_path="data/processed" \
		--base_model_name=${base_model_name} \
		--weight_decay=${weight_decay} \
		--learning_rate=${learning_rate} \
		--train_per_device_batch_size=${train_per_device_batch_size} \
		--test_per_device_batch_size=${test_per_device_batch_size} \
		--num_epochs=${num_epochs} \
		--train_dataloader_prng_key=${train_dataloader_prng_key} \
		--test_every_num_steps=${test_every_num_steps}
		

setup_test_data: 
	rm data/testing/raw/train/pos/* data/testing/raw/train/neg/*
	FILES="$(shell ls -d data/raw/train/pos/* | head -n 10)"; cp $$FILES data/testing/raw/train/pos
	FILES="$(shell ls -d data/raw/train/neg/* | head -n 17)"; cp $$FILES data/testing/raw/train/neg

run_tests:
	python3 -m unittest src.tests