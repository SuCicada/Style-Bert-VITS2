CONDA_ENV=Style-Bert-VITS2

run:
	$(conda_run) python app.py --dir model_assets \
		--server_name 0.0.0.0
