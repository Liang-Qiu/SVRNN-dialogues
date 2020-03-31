.PHONY: install dataset style clean

install:
	pip install -r requirements.txt

dataset:
	# python data_apis/ubuntu_dataset/src/dataset_generator.py --data=$(data_path)
	# python data_apis/ubuntu_dataset/src/dataset_separator.py --data=$(data_path)/#ubuntu.gz
	python data_apis/ubuntu_dataset/src/sample_generator.py --train_data=$(data_path)/train-data.gz --dev_data=$(data_path)/dev-data.gz --test_data=$(data_path)/test-data.gz

style:
	yapf -i -r --style google .

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf docs/_build
	rm -rf .pytest_cache/