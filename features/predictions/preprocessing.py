from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

def preprocess_player_data(df):
    """Preprocess the player data for modeling"""
    
    # 1. Sort and filter
    df = df.sort_values(["player_id", "date"])
    df = df[     # Keep rows where team_id matches t1 or t2 OR where both t1 and t2 are missing
        (df["team_id"] == df["t1"]) |
        (df["team_id"] == df["t2"]) |
        (df["t1"].isna() & df["t2"].isna())
    ]

    # Convert date columns to datetime
    df["date"] = pd.to_datetime(df["date"])
    df["md"] = pd.to_datetime(df["md"])

    # 2. Date and matchday calculations 
    df["next_day"] = df.groupby("player_id")["date"].shift(-1) 
    df["next_md"] = df.groupby("player_id")["md"].transform(
        lambda x: x.shift(-1).where(x.shift(-1) != x).bfill()
    )
    df["days_to_next"] = (df["next_md"] - df["date"]).dt.days

    # 3. Next day market value
    df["mv_next_day"] = df.groupby("player_id")["mv"].shift(-1)
    df["mv_target"] = df["mv_next_day"] - df["mv"]
    df = df[df["mv"] != 0.0]

    # 4. Feature engineering 
    # Market value trend 1d
    df["mv_change_1d"] = df["mv"] - df.groupby("player_id")["mv"].shift(1)
    df["mv_trend_1d"] = df.groupby("player_id")["mv"].pct_change(fill_method=None)
    df["mv_trend_1d"] = df["mv_trend_1d"].replace([np.inf, -np.inf], 0).fillna(0)

    # Market value trend 3d
    df["mv_change_3d"] = df["mv"] - df.groupby("player_id")["mv"].shift(3)
    df["mv_vol_3d"] = df.groupby("player_id")["mv"].rolling(3).std().reset_index(0,drop=True)

    # Market value trend 7d
    df["mv_trend_7d"] = df.groupby("player_id")["mv"].pct_change(periods=7, fill_method=None)
    df["mv_trend_7d"] = df["mv_trend_7d"].replace([np.inf, -np.inf], 0).fillna(0)

    ## League-wide market context
    df["market_divergence"] = (df["mv"] / df.groupby("md")["mv"].transform("mean")).rolling(3).mean()

    # 5. Clip outliers in mv_target
    Q1 = df["mv_target"].quantile(0.25)
    Q3 = df["mv_target"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR

    df["mv_target_clipped"] = df["mv_target"].clip(lower_bound, upper_bound)

    # 6. Fill missing values
    df = df.fillna({
        "market_divergence": 1,
        "mv_change_3d": 0,
        "mv_vol_3d": 0,
        "p": 0,
        "ppm": 0,
        "won": -1
    })

    # 7. Cutout todays values and store them
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    max_date = (now - timedelta(days=1)) if now <= cutoff_time else now
    max_date = max_date.date()

    today_df = df[df["date"].dt.date >= max_date]

    # Drop those values from today from df
    df = df[df["date"].dt.date < max_date]

    # 8. Drop rows with NaN in critical columns
    df = df.dropna(subset=["mv_change_1d", "next_day", "next_md", "days_to_next", "mv_next_day", "mv_target", "mv_target_clipped"])

    return df, today_df


def split_data(df, features, target):
    """Split the data into training and testing sets based on date to avoid data leakage"""

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.75)
    split_date = df["date"].iloc[split_idx]

    # Split by time, to avoid data leakage
    train = df[df["date"] < split_date]
    test = df[(df["date"] >= split_date)]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    return X_train, X_test, y_train, y_test


def prepare_horizon_forecast_data(df, features, horizon_days):
    """
    Prepare training data for a cumulative market value prediction
    over an exact number of future daily market value updates.
    """

    if horizon_days <= 0:
        return pd.DataFrame()

    data = df.copy()

    # Normalize dates and keep one row per player/date
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = (
        data.sort_values(["player_id", "date"])
        .drop_duplicates(subset=["player_id", "date"], keep="last")
    )

    # Lookup table containing future market values
    future_values = data[["player_id", "date", "mv"]].copy()
    future_values = future_values.rename(
        columns={
            "date": "future_date",
            "mv": "future_mv"
        }
    )

    # Exact future date corresponding to our forecast horizon
    data["future_date"] = (
        data["date"] + pd.to_timedelta(horizon_days, unit="D")
    )

    # Attach the market value from that future date
    data = data.merge(
        future_values,
        on=["player_id", "future_date"],
        how="left"
    )

    # Cumulative market value change over the entire horizon
    data["mv_target_horizon"] = data["future_mv"] - data["mv"]

    # Remove rows where no exact future market value is available
    data = data.dropna(
        subset=features + ["mv_target_horizon"]
    )

    if data.empty:
        return data

    # Clip extreme target outliers, analogous to the existing daily model
    q1 = data["mv_target_horizon"].quantile(0.25)
    q3 = data["mv_target_horizon"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 2.5 * iqr
    upper_bound = q3 + 2.5 * iqr

    data["mv_target_horizon_clipped"] = (
        data["mv_target_horizon"].clip(
            lower_bound,
            upper_bound
        )
    )

    return data

def estimate_market_momentum_decay(
    df,
    min_abs_change=50_000
):
    """
    Estimate how much of a player's daily market value momentum
    is typically retained on the following day.

    A factor of 0.95 means:
    tomorrow's change is historically about 95% of today's change.

    Positive and negative trends are estimated separately.
    """

    data = df[
        [
            "player_id",
            "date",
            "mv_change_1d"
        ]
    ].copy()

    data["date"] = (
        pd.to_datetime(data["date"])
        .dt.normalize()
    )

    data = (
        data
        .dropna(
            subset=[
                "player_id",
                "date",
                "mv_change_1d"
            ]
        )
        .sort_values(
            ["player_id", "date"]
        )
        .drop_duplicates(
            subset=["player_id", "date"],
            keep="last"
        )
    )

    # Create a copy containing the following day's change
    next_day = data[
        [
            "player_id",
            "date",
            "mv_change_1d"
        ]
    ].copy()

    next_day = next_day.rename(
        columns={
            "mv_change_1d":
                "next_mv_change"
        }
    )

    # Shift the next-day date backwards so that
    # current day and following day can be merged
    next_day["date"] = (
        next_day["date"]
        - pd.Timedelta(days=1)
    )

    pairs = data.merge(
        next_day,
        on=["player_id", "date"],
        how="inner"
    )

    # Ignore tiny daily changes.
    # They are not useful for estimating momentum persistence.
    pairs = pairs[
        pairs["mv_change_1d"].abs()
        >= min_abs_change
    ]

    def calculate_factor(subset):

        if len(subset) == 0:
            return np.nan, 0

        x = subset[
            "mv_change_1d"
        ].to_numpy(dtype=float)

        y = subset[
            "next_mv_change"
        ].to_numpy(dtype=float)

        denominator = np.dot(x, x)

        if denominator == 0:
            return np.nan, len(subset)

        # Linear regression through the origin:
        # next_change = factor * current_change
        factor = (
            np.dot(x, y)
            / denominator
        )

        return factor, len(subset)

    positive_factor, positive_samples = (
        calculate_factor(
            pairs[
                pairs["mv_change_1d"] > 0
            ]
        )
    )

    negative_factor, negative_samples = (
        calculate_factor(
            pairs[
                pairs["mv_change_1d"] < 0
            ]
        )
    )

    return {
        "positive_factor":
            positive_factor,
        "negative_factor":
            negative_factor,
        "positive_samples":
            positive_samples,
        "negative_samples":
            negative_samples
    }

def estimate_regime_horizon_decay(
    df,
    horizon_days,
    min_change=100_000
):
    """
    Estimate multi-day market value persistence for historical
    situations similar to the current pre-matchday momentum phase.

    Instead of extrapolating a one-day factor repeatedly, this
    measures the actual cumulative change over the entire horizon.
    """

    if horizon_days <= 1:
        return {}

    data = df.copy()

    data["date"] = (
        pd.to_datetime(data["date"])
        .dt.normalize()
    )

    data = (
        data
        .sort_values(["player_id", "date"])
        .drop_duplicates(
            subset=["player_id", "date"],
            keep="last"
        )
    )

    # Exact market value horizon_days later
    future_values = data[
        ["player_id", "date", "mv"]
    ].copy()

    future_values = future_values.rename(
        columns={
            "date": "future_date",
            "mv": "future_mv"
        }
    )

    data["future_date"] = (
        data["date"]
        + pd.to_timedelta(
            horizon_days,
            unit="D"
        )
    )

    data = data.merge(
        future_values,
        on=["player_id", "future_date"],
        how="left"
    )

    data["cumulative_future_change"] = (
        data["future_mv"]
        - data["mv"]
    )

    required_columns = [
        "mv_change_1d",
        "mv_change_3d",
        "mv_trend_7d",
        "days_to_next",
        "cumulative_future_change"
    ]

    data = data.dropna(
        subset=required_columns
    )

    # We want historical situations reasonably similar
    # to the current long pre-matchday window.
    min_days_to_match = max(
        7,
        horizon_days - 2
    )

    def implied_decay(multiplier):

        # A normal geometric decay with phi between
        # 0 and 1 has a cumulative multiplier
        # between 1 and horizon_days.
        if (
            pd.isna(multiplier)
            or multiplier <= 1
        ):
            return np.nan

        if multiplier >= horizon_days:
            return 1.0

        low = 0.0
        high = 1.0

        for _ in range(60):

            phi = (
                low + high
            ) / 2

            if phi == 1:
                geometric_sum = horizon_days

            else:
                geometric_sum = (
                    1 - phi ** horizon_days
                ) / (
                    1 - phi
                )

            if geometric_sum < multiplier:
                low = phi
            else:
                high = phi

        return (
            low + high
        ) / 2

    def summarize(subset):

        subset = subset.copy()

        if subset.empty:
            return {
                "samples": 0,
                "median_multiplier": np.nan,
                "implied_decay": np.nan
            }

        # Relative cumulative development compared with
        # the known current 24h change.
        subset["horizon_multiplier"] = (
            subset["cumulative_future_change"]
            / subset["mv_change_1d"]
        )

        # Median is deliberately used because individual
        # market value paths can contain extreme reversals.
        multiplier = (
            subset["horizon_multiplier"]
            .median()
        )

        return {
            "samples": len(subset),
            "median_multiplier": multiplier,
            "implied_decay": implied_decay(
                multiplier
            )
        }

    # Positive momentum:
    # rising today, rising across 3 days and 7 days,
    # and relatively far away from the next match.
    positive_persistent = data[
        (data["mv_change_1d"] >= min_change)
        & (data["mv_change_3d"] > 0)
        & (data["mv_trend_7d"] > 0)
        & (
            data["days_to_next"]
            >= min_days_to_match
        )
    ]

    # Stronger definition:
    # not just positive, but a substantial multi-day streak.
    positive_strong = data[
        (data["mv_change_1d"] >= 150_000)
        & (
            data["mv_change_3d"]
            >= 2 * data["mv_change_1d"]
        )
        & (data["mv_trend_7d"] > 0)
        & (
            data["days_to_next"]
            >= min_days_to_match
        )
    ]

    negative_persistent = data[
        (data["mv_change_1d"] <= -min_change)
        & (data["mv_change_3d"] < 0)
        & (data["mv_trend_7d"] < 0)
        & (
            data["days_to_next"]
            >= min_days_to_match
        )
    ]

    negative_strong = data[
        (data["mv_change_1d"] <= -150_000)
        & (
            data["mv_change_3d"]
            <= 2 * data["mv_change_1d"]
        )
        & (data["mv_trend_7d"] < 0)
        & (
            data["days_to_next"]
            >= min_days_to_match
        )
    ]

    return {
        "positive_persistent":
            summarize(positive_persistent),

        "positive_strong":
            summarize(positive_strong),

        "negative_persistent":
            summarize(negative_persistent),

        "negative_strong":
            summarize(negative_strong)
    }
