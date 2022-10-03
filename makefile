tests:
	pytest tests

formatting:
	black image_retrieval/
	isort image_retrieval/

notebook-sync:
	jupytext --sync  notebooks/*.ipynb


clean:
	rm -rf lightning_logs
	rm -rf checkpoints

tensorboard:
	tensorboard --logdir ./lightning_logs
