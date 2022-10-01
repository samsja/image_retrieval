tests:
	pytest tests

formatting:
	black image_retrieval/
	isort image_retrieval/

notebook-sync:
	jupytext --sync  notebooks/*.ipynb


clean_log:
	rm -rf lightning_logs

tensorboard:
	tensorboard --logdir ./lightning_logs
