# graph-matching

## Install

```bash
make install
```

## Prepare Dataset

Download [Ubuntu Chat Corpus](https://daviduthus.org/UCC/)

```bash
cd data/ubuntu_dataset/src
python dataset_generator.py --data=path/to/your/ubuntu/chat/corpus/#ubuntu
python dataset_seperator.py --data=path/to/your/ubuntu/chat/corpus/#ubuntu/#ubuntu.gz
python sample_generator.py --train_data=path/to/your/ubuntu/chat/corpus/#ubuntu/train-data.gz --dev_data=path/to/your/ubuntu/chat/corpus/#ubuntu/dev-data.gz --test_data=path/to/your/ubuntu/chat/corpus/#ubuntu/test-data.gz
```

## Develop

```bash
make develop
```

## Test

Run all tests in the `/tests` folder

```bash
make develop
make test
```

Test coverage

```bash
pytest --cov=graph_matching tests/
```
