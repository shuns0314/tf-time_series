import warnings

import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_probability as tfp

from tensorflow_probability import sts

# Allow external control of optimization to reduce test runtimes.
num_variational_steps = 10  # @param { isTemplate: true}
num_variational_steps = int(num_variational_steps)

optimizer = tf.optimizers.Adam(learning_rate=0.1)


def build_model(observed_time_series):
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    seasonal = tfp.sts.Seasonal(
        num_seasons=12, observed_time_series=observed_time_series
    )
    model = sts.Sum(
        [trend, seasonal], observed_time_series=observed_time_series
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


def main():
    df = pd.read_csv("data/sample.csv")
    data = np.array(df["y"])
    model = build_model(data)
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
        model=model
    )
    train(data, model, variational_posteriors)
    # Draw samples from the variational posterior.
    sample = variational_posteriors.sample(50)
    steps = 100
    forecast(model, data, sample, steps)


if __name__ == "__main__":
    main()
