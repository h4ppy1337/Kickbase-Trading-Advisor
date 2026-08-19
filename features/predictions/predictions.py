from kickbase_api.league import get_league_players_on_market
from kickbase_api.user import get_players_in_squad
from kickbase_api.manager import get_managers, get_manager_squad
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

def live_data_predictions(today_df, model, features):
    """Make live data predictions for today_df using the trained model"""

    # Set features and copy df
    today_df_features = today_df[features]
    today_df_results = today_df.copy()

    # Predict mv_target
    today_df_results["predicted_mv_target"] = np.round(model.predict(today_df_features), 2)

    # Sort by predicted_mv_target descending
    today_df_results = today_df_results.sort_values("predicted_mv_target", ascending=False)

    # Filter date to today or yesterday if before 22:15, because mv is updated around 22:15
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    date = (now - timedelta(days=1)) if now <= cutoff_time else now
    date = date.date()

    # Drop rows where NaN mv
    today_df_results = today_df_results.dropna(subset=["mv"])

    # Keep only relevant columns
    today_df_results = today_df_results[["player_id", "first_name", "last_name", "position", "team_name", "date", "mv_change_1d", "mv_trend_1d", "mv", "predicted_mv_target"]]

    return today_df_results


def join_current_squad(token, league_id, today_df_results):
    squad_players = get_players_in_squad(token, league_id)

    squad_df = pd.DataFrame(squad_players["it"])

    # Join squad_df ("i") with today_df ("player_id")
    squad_df = (
        pd.merge(today_df_results, squad_df, left_on="player_id", right_on="i")
        .drop(columns=["i"])
    )

    # Rename prob to s_11_prob for better understanding
    if "prob" not in squad_df.columns:
        squad_df["prob"] = np.nan  # Placeholder for non-pro users
    squad_df = squad_df.rename(columns={"prob": "s_11_prob"})

    # Rename mv_change_1d to mv_change_yesterday for better understanding
    squad_df = squad_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})

    # Rename "mv_x" to "mv" for better understanding
    squad_df = squad_df.rename(columns={"mv_x": "mv"})

    # Keep only relevant columns
    squad_df = squad_df[["last_name", "team_name", "mv", "mv_change_yesterday", "predicted_mv_target", "s_11_prob"]]

    return squad_df 


# TODO Add fail-safe check before player expires if the prob (starting 11) is still high, so no injuries or anything. if it dropped. dont bid / reccommend
def join_current_market(token, league_id, today_df_results):
    """Join the live predictions with the current market data to get bid recommendations"""

    players_on_market = get_league_players_on_market(token, league_id)

    # players_on_market to DataFrame
    market_df = pd.DataFrame(players_on_market)

    # Join market_df ("id") with today_df ("player_id")
    bid_df = (
        pd.merge(today_df_results, market_df, left_on="player_id", right_on="id")
        .drop(columns=["id"])
    )

    # exp contains seconds until expiration
    bid_df["hours_to_exp"] = np.round((bid_df["exp"] / 3600), 2)

    # check if current sysdate + hours_to_exp is after the next 22:00
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    next_22 = now.replace(hour=22, minute=0, second=0, microsecond=0)
    diff = np.round((next_22 - now).total_seconds() / 3600, 2)

    # If hours_to_exp < diff then it expires today
    bid_df["expiring_today"] = bid_df["hours_to_exp"] < diff

    # Drop rows where predicted_mv_target is less than 5000
    bid_df = bid_df[bid_df["predicted_mv_target"] > 5000]

    # Sort by predicted_mv_target descending
    bid_df = bid_df.sort_values("predicted_mv_target", ascending=False)

    # Rename prob to s_11_prob for better understanding
    if "prob" not in bid_df.columns:
        bid_df["prob"] = np.nan  # Placeholder for non-pro users
    bid_df = bid_df.rename(columns={"prob": "s_11_prob"})

    # Rename mv_change_1d to mv_change_yesterday for better understanding
    bid_df = bid_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})

    # Keep only relevant columns
    bid_df = bid_df[["last_name", "team_name", "mv", "mv_change_yesterday", "predicted_mv_target", "s_11_prob", "hours_to_exp", "expiring_today"]]

    return bid_df


def live_horizon_predictions(today_df, model, features, horizon_days):
    """
    Predict the cumulative market value change for every player
    until the next matchday.
    """

    current_df = (
        today_df
        .sort_values(["player_id", "date"])
        .drop_duplicates(subset=["player_id"], keep="last")
        .copy()
    )

    # No market value update left before kickoff
    if horizon_days <= 0:
        current_df["predicted_mv_change_until_matchday"] = 0.0
    else:
        prediction_features = current_df[features]

        current_df["predicted_mv_change_until_matchday"] = np.round(
            model.predict(prediction_features),
            2
        )

    return current_df[
        [
            "player_id",
            "first_name",
            "last_name",
            "team_name",
            "mv",
            "predicted_mv_change_until_matchday"
        ]
    ]


def build_manager_value_forecast(
    token,
    league_id,
    manager_budgets_df,
    player_forecast_df,
    future_login_bonus
):
    """
    Aggregate player market value predictions for every manager
    and calculate projected total manager value at next matchday.
    """

    predictions = player_forecast_df.copy()
    predictions["player_id"] = predictions["player_id"].astype(str)

    prediction_lookup = dict(
        zip(
            predictions["player_id"],
            predictions["predicted_mv_change_until_matchday"]
        )
    )

    # Normalize manager names for matching
    budget_df = manager_budgets_df.copy()
    budget_df["User_Key"] = (
        budget_df["User"].astype(str).str.strip()
    )

    managers = get_managers(token, league_id)

    results = []

    for manager_name, manager_id in managers:

        squad_data = get_manager_squad(
            token,
            league_id,
            manager_id
        )

        squad_players = squad_data.get("it", [])

        predicted_change = 0.0
        predicted_players = 0

        for player in squad_players:

            player_id = str(player.get("pi"))

            if player_id in prediction_lookup:
                change = prediction_lookup[player_id]

                if pd.notna(change):
                    predicted_change += float(change)
                    predicted_players += 1

        manager_key = str(manager_name).strip()

        manager_budget_row = budget_df[
            budget_df["User_Key"] == manager_key
        ]

        if manager_budget_row.empty:
            print(
                f"Warning: No budget information found for "
                f"{manager_name}"
            )
            continue

        current_team_value = float(
            manager_budget_row.iloc[0]["Team Value"]
        )

        current_budget = float(
            manager_budget_row.iloc[0]["Budget"]
        )

        predicted_team_value = (
            current_team_value + predicted_change
        )

        predicted_budget = (
            current_budget + future_login_bonus
        )

        manager_value = (
            predicted_team_value + predicted_budget
        )

        squad_size = len(squad_players)

        coverage = (
            f"{predicted_players}/{squad_size}"
            if squad_size > 0
            else "0/0"
        )

        results.append({
            "User": manager_name,
            "Team Value Now": current_team_value,
            "Predicted MV Change": predicted_change,
            "Team Value @ MD": predicted_team_value,
            "Cash Now": current_budget,
            "Future Login Bonus": future_login_bonus,
            "Cash @ MD": predicted_budget,
            "Manager Value @ MD": manager_value,
            "Prediction Coverage": coverage
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values(
            "Manager Value @ MD",
            ascending=False,
            ignore_index=True
        )

    return result_df
