CONDA_ENV=Style-Bert-VITS2

run:
	$(conda_run) python app.py $(args)

docker-run-local:
	cd deploy/local && \
		docker-compose down && \
		docker-compose up -d
