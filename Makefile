.PHONY: install get_data develop style clean test

install:
	pip install -r requirements.txt

train:
	python main.py

decode:
	python main/py --forward_only True

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