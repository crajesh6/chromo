import tensorflow as tf
from wandb.keras import WandbCallback
from ray.tune.integration.keras import TuneReportCallback

from chromo import model_zoo, utils, chrombpnet_utils


class ModelBuilder:
    def __init__(self, config, epochs, batch_size, log_wandb, tuning_mode, sigma=0.001):
        self.config = config
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_wandb = log_wandb
        self.tuning_mode = tuning_mode
        self.tune_metrics = None
        self.sigma = sigma

        # Initialize the dictionary mapping
        self.model_functions = {
            "chrombpnet": chrombpnet_utils.chrombpnet,
            "factorized_homininn": chrombpnet_utils.factorized_homininn,
            "default": model_zoo.hominin_base_model,
        }

    def build_model(self, model_fn):
        print("Building model...")
        # Fetch the model function from the dictionary, use 'default' if model_fn not found
        model_function = self.model_functions.get(
            model_fn, self.model_functions["default"]
        )
        # Call the model function with unpacked config
        model = model_function(**self.config)
        return model

    def _get_early_stopping_callback(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )

    def _get_reduce_lr_callback(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            mode="min",
            verbose=1,
        )

    def _get_tune_report_callback(self):
        return TuneReportCallback({key: key for key in self.tune_metrics})

    def _set_tune_report_callback(self, metrics):
        self.tune_metrics = metrics

    def compile_model(self, model, dataset):
        if dataset == "chrombpnet":  # already compiled
            return model

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = "mse"
        metrics = [utils.Spearman, utils.pearson_r]
        tune_metrics = ["val_pearson_r"]
        self._set_tune_report_callback(tune_metrics)

        if dataset == "scbasset":
            auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")
            aupr = tf.keras.metrics.AUC(curve="PR", name="aupr")
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=False, label_smoothing=0.0
            )
            metrics = [auroc, aupr]
            tune_metrics = ["val_aupr"]
            self._set_tune_report_callback(tune_metrics)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def fit_model(self, model, train_data, valid_data, dataset):
        callbacks = [
            self._get_early_stopping_callback(),
            self._get_reduce_lr_callback(),
        ]

        if self.log_wandb:
            callbacks += [WandbCallback(save_model=False)]
        if self.tuning_mode:
            tune_report_callback = self._get_tune_report_callback()
            callbacks.append(tune_report_callback)

        if dataset == "plantstarr":
            x_train, y_train = train_data
            history = model.fit(
                x_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_split=0.1,
                callbacks=callbacks,
            )
        elif dataset == "deepstarr":
            x_train, y_train = train_data
            x_valid, y_valid = valid_data
            history = model.fit(
                x_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(x_valid, y_valid),
                callbacks=callbacks,
            )
        elif dataset == "scbasset":
            x_train, y_train = train_data
            x_valid, y_valid = valid_data
            history = model.fit(
                x_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(x_valid, y_valid),
                callbacks=callbacks,
            )
        elif dataset == "hepg2":
            x_train, y_train = train_data
            x_valid, y_valid = valid_data
            history = model.fit(
                x_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(x_valid, y_valid),
                callbacks=callbacks,
            )
        elif dataset == "chrombpnet":
            history = model.fit(
                train_data,
                validation_data=valid_data,
                epochs=self.epochs,
                verbose=1,
                callbacks=callbacks,
            )

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return model, history
