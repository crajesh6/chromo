import os
import sys

import click

from chromo import run_chromo


@click.command()
@click.option("--config_file", type=str)
@click.option("--dataset_name", type=str, default="deepstarr")
@click.option("--gpu", type=str, default=None)
@click.option("--smoke_test", type=bool, default=False)
@click.option("--log_wandb", type=bool, default=True)
def main(
    config_file: str, dataset_name: str, gpu: str, smoke_test: bool, log_wandb: bool
):
    # check if dataset does not match experiment
    if dataset_name not in config_file:
        print("Dataset and experiment do not match. Abort!")
        sys.exit()

    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    epochs = 2 if smoke_test else 100

    save_path = config_file.split("config.yaml")[0]
    config = run_chromo.load_config(config_file)

    tuner = run_chromo.chromoTuner(
        config,
        epochs=epochs,
        tuning_mode=False,
        save_path=save_path,
        dataset=dataset_name,
        subsample=smoke_test,
        log_wandb=log_wandb,
    )

    # train model
    tuner.train_model()

    # # load model weights:
    #
    #
    # # interpret model
    # print("Evaluating first conv. layer!")
    # tuner.visualize_filters(layer=layer)
    #
    # # print("Calculating saliency maps!")
    # tuner.calculate_saliency_maps(class_index=0)
    # tuner.calculate_saliency_maps(class_index=1)
    #
    # # run GLIFAC
    # print("Running GLIFAC!")
    # tuner.run_glifac(class_index=0)
    # tuner.run_glifac(class_index=1)


if __name__ == "__main__":
    main()
