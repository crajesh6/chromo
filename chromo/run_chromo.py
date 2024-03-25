import os
import yaml
import pandas as pd
from pathlib import Path
import wandb

from chromo import utils, data_processor, model_builder


os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.full_load(file)  # Changed full_load to safe_load
    return config


class chromoTuner:
    def __init__(
        self,
        config,
        epochs=60,
        batch_size=64,
        tuning_mode=False,
        save_path="default_save_path",
        subsample=False,
        dataset="chrombpnet",
        log_wandb=True,
        wandb_project=None,
        wandb_name=None,
        magic=None,
        model_fn="factorized_homininn",
    ):
        self.config = config
        self.subsample = subsample
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.tuning_mode = tuning_mode
        self.save_path = save_path
        self.wandb_project = wandb_project or f"chromo_{self.dataset}"
        self.wandb_name = wandb_name or self.save_path
        self.log_wandb = log_wandb
        self.magic = magic  # Assumed it's used somewhere not shown in the snippet
        self.data_processor = data_processor.DataProcessor(
            dataset=self.dataset, subsample=self.subsample
        )
        self.mb = model_builder.ModelBuilder(
            self.config, self.epochs, self.batch_size, self.log_wandb, self.tuning_mode
        )
        self.model_fn = model_fn

    def update_config(self, key, value):
        self.config[key] = value
        self.mb.config = self.config

    def train_model(self):
        if self.log_wandb:
            wandb.init(
                project=self.wandb_project, name=self.wandb_name, config=self.config
            )

        train_data, valid_data, test_data = (
            self.data_processor.load_all_data()
        )  # Assuming there's a method to load all data at once for efficiency

        model = self.mb.build_model(
            self.model_fn
        )  # Assuming build_model method takes model_fn as an argument
        model, history = self.compile_and_train_model(model, train_data, valid_data)
        pd.DataFrame(history.history).to_csv(
            os.path.join(self.save_path, "history.csv")
        )

        self.save_model(model)

        with open(os.path.join(self.save_path, "config.yaml"), "w") as file:
            yaml.dump(self.config, file)

        self.evaluate_model()
        print("Done training and evaluating the model!")

    def compile_and_train_model(self, model, train_data, valid_data):
        model = self.mb.compile_model(model, self.dataset)
        model, history = self.mb.fit_model(model, train_data, valid_data)
        return model, history

    def evaluate_model(self):
        model = self.mb.load_and_compile_model(
            self.dataset, self.save_path
        )  # Assuming there's a method for loading and compiling

        print("Evaluating model!")
        evaluation_path = f"{self.save_path}/evaluation"
        Path(evaluation_path).mkdir(parents=True, exist_ok=True)

        evaluation_function = None

        if self.dataset == "deepstarr":
            evaluation_function = utils.evaluate_model_deepstarr
        elif self.dataset == "plantstarr":
            evaluation_function = utils.evaluate_model_plantstarr
        elif self.dataset == "scbasset":
            evaluation_function = utils.evaluate_model_scbasset

        elif self.dataset == "hepg2":
            evaluation_function = utils.evaluate_model_hepg2

        elif self.dataset == "chrombpnet":
            evaluation_function = utils.evaluate_model_chrombpnet


        if evaluation_function:
            test_data = self.data_processor.load_data("test")
            df = evaluation_function(
                model, test_data, os.path.join(self.save_path, "evaluation")
            )
            return df
        else:
            return None

    def save_model(self, model):
        if self.dataset == "chrombpnet":
            model.save(os.path.join(self.save_path, "model"), save_format="tf")
        else:
            model.save_weights(os.path.join(self.save_path, "weights"))
