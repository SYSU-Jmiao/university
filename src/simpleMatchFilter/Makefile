test:
	nosetests ./tests

vector_based:
	 python vector_based_locator.py ./sample_data/antenna_samples/original/ ./sample_data/antenna_samples/filterd/


init_db:
	docker-compose up -d db

tear_down:
	docker-compose kill db
	docker-compose rm -f
