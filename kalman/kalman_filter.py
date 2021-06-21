from pathlib import Path
from os import makedirs
import pickle
from typing import Any

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow_probability import sts

# Allow external control of optimization to reduce test runtimes.
num_variational_steps = 200  # @param { isTemplate: true}
num_variational_steps = int(num_variational_steps)

optimizer = tf.optimizers.Adam(learning_rate=0.1)
tf.config.optimizer.set_jit(True)


def build_model(observed_time_series):
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    weekly_seasonal = tfp.sts.Seasonal(
        num_seasons=7,
        observed_time_series=observed_time_series,
        name="day_of_week_effect",
    )
    month_seasonal = tfp.sts.Seasonal(
        num_seasons=4,
        num_steps_per_season=7,
        observed_time_series=observed_time_series,
        name="month_of_week_effect",
    )
    autoregressive = sts.Autoregressive(
        order=1,
        observed_time_series=observed_time_series,
        name="autoregressive",
    )
    model = sts.Sum(
        [trend, weekly_seasonal, month_seasonal, autoregressive],
        observed_time_series=observed_time_series,
    )
    return model


def train(data, model, variational_posteriors):
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_log_prob(observed_time_series=data),
        surrogate_posterior=variational_posteriors,
        optimizer=optimizer,
        num_steps=num_variational_steps,
    )
    return elbo_loss_curve


def forecast(model, data, samples, steps):
    forecast_dist = tfp.sts.forecast(
        model,
        observed_time_series=data,
        parameter_samples=samples,
        num_steps_forecast=steps,
    )
    return forecast_dist


def plot(x, y, forcast, std, step):
    plt.plot(x, y)
    print(x)
    _x = pd.date_range(x.iloc[-1], periods=step, freq="1 d")
    plt.plot(_x, forcast)
    plt.fill_between(
        _x,
        forcast + std,
        forcast - std,
        facecolor="y",
        alpha=0.3,
    )
    plt.savefig("tes.png")


def main():
    df = pd.read_csv("data/sample.csv")
    x = pd.to_datetime(df["ds"])
    y = np.array(df["y"])

    model = build_model(y)
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
        model=model
    )
    train(y, model, variational_posteriors)
    # Draw samples from the variational posterior.
    sample = variational_posteriors.sample(50)
    steps = 100
    pred = forecast(model, y, sample, steps)
    plot(
        x, y, pred.mean().numpy()[..., 0], pred.stddev().numpy()[..., 0], steps
    )


if __name__ == "__main__":
    main()
