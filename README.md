# Important
This project uses `git lfs`.
You will need to:
- clone this repository
- change dir to repo dir
- `git lfs pull`

# Startup Guide

```
pip install -r requirements.txt
pip install .
```
Then edit config.yaml to set default params for models etc.

### For training use:
```
make train
```

### For eval use:
```
make eval \
ARGS="--model_path='../../reports/mlruns/0/{write your runs}/artifacts/model/data/model.pth'"
```
### Example:
```
make eval \
ARGS="--model_path='../../reports/mlruns/0/5ad1393c966543be89e1f8cb6bd14397/artifacts/model/data/model.pth' --num_classes=3"
```

### Running mlflow:
```
make mlflow
```

# Recommended Requirements
- Python 3.9.*
- NVIDIA GPU
- Installed Make

### For Ubuntu:
```
sudo apt install build-essential 
make -version
```

### For Windows:
Install mingw, then use `mingw32-make` instead `make` in console.