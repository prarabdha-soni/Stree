.PHONY: format

format:
	isort generate.py stree
	yapf -i -r *.py generate.py stree
