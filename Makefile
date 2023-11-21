install:
	pip install -r requirements.txt

train:
	cd src/cli && python train_cli_commands.py $(ARGS)

eval:
	cd src/cli && python test_cli_commands.py $(ARGS)

mlflow:
	mlflow ui --backend-store-uri reports/mlruns/
