import click
import yaml
from typing import Dict, Any
from src.models.eval_model import eval_model


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_from_config(config: Dict[str, Any], param_name: str) -> Any:
    return config.get(param_name, 0)


config_path = '../../config.yaml'
config_params = load_config(config_path)


@click.command()
@click.option('--model_path', required=True, type=click.Path())
@click.option('--image_dir', default=get_default_from_config(config_params, 'image_dir'), type=click.Path(exists=True))
@click.option('--num_classes', default=get_default_from_config(config_params, 'num_classes'), type=click.INT)
@click.option('--batch_size', default=get_default_from_config(config_params, 'batch_size'), help='Batch size', type=click.INT)
@click.option('--val_proportion', default=get_default_from_config(config_params, 'val_proportion'), type=click.FLOAT)
@click.option('--test_proportion', default=get_default_from_config(config_params, 'test_proportion'), type=click.FLOAT)
def get_cli_params_for_eval(model_path,
                            image_dir,
                            num_classes,
                            batch_size,
                            val_proportion,
                            test_proportion,):
    params = locals()
    eval_model(**params)


if __name__ == '__main__':
    get_cli_params_for_eval()
