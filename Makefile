travis: travis-doc travis-unit travis-examples travis-end

travis-install:
	pip install -e .
	pip install -r requirements.txt
	pip install -r docs/requirements.txt
	pip install -r tests/requirements.txt

travis-doc: build-doc

travis-unit:
	COVERAGE_FILE=.coverage.doc1 coverage run --parallel-mode -m doctest sspace/space.py
	COVERAGE_FILE=.coverage.doc2 coverage run --parallel-mode -m doctest sspace/conditionals.py
	COVERAGE_FILE=.coverage.unit coverage run --parallel-mode -m pytest --cov=sspace tests

travis-examples:
	COVERAGE_FILE=.coverage.simple coverage run examples/simple.py

travis-end:
	coverage combine
	coverage report -m
	coverage xml
	codecov

build-doc:
	# sphinx-apidoc -e -o docs/api olympus
	sphinx-build -W --color -c docs/ -b html docs/ _build/html

serve-doc:
	sphinx-serve

update-doc: build-doc serve-doc