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


def live_damped_predictions(
    live_predictions_df,
    horizon_days,
    positive_decay_factor,
    negative_decay_factor
):
    """
    Extend the existing next-update ML prediction until the next
    matchday using different exponential decay factors for rising
    and falling players.
    """

    if not 0 < positive_decay_factor <= 1:
        raise ValueError(
            "positive_decay_factor must be greater than 0 and at most 1."
        )

    if not 0 < negative_decay_factor <= 1:
        raise ValueError(
            "negative_decay_factor must be greater than 0 and at most 1."
        )

    results = live_predictions_df.copy()

    # Existing Random Forest prediction for the next MV update
    results["predicted_next_update"] = (
        results["predicted_mv_target"]
    )

    # Use the positive decay for predicted risers and
    # the negative decay for predicted fallers.
    results["decay_factor_used"] = np.where(
        results["predicted_next_update"] >= 0,
        positive_decay_factor,
        negative_decay_factor
    )

    if horizon_days <= 0:

        results["predicted_last_update"] = 0.0
        results[
            "predicted_mv_change_until_matchday"
        ] = 0.0

    else:

        decay = results["decay_factor_used"]

        # Geometric sum:
        # 1 + phi + phi^2 + ... + phi^(n-1)
        cumulative_multiplier = np.where(
            decay == 1,
            float(horizon_days),
            (
                1 - decay ** horizon_days
            ) / (
                1 - decay
            )
        )

        last_update_multiplier = (
            decay ** (horizon_days - 1)
        )

        results["predicted_last_update"] = np.round(
            results["predicted_next_update"]
            * last_update_multiplier,
            2
        )

        results[
            "predicted_mv_change_until_matchday"
        ] = np.round(
            results["predicted_next_update"]
            * cumulative_multiplier,
            2
        )

    return results[
        [
            "player_id",
            "first_name",
            "last_name",
            "team_name",
            "mv",
            "mv_change_1d",
            "predicted_next_update",
            "predicted_last_update",
            "predicted_mv_change_until_matchday",
            "decay_factor_used"
        ]
    ]


def build_manager_value_forecast(
    token,
    league_id,
    manager_budgets_df,
    player_forecast_df,
    future_login_bonus,
    market_updates_until_matchday,
    positive_decay_factor,
    negative_decay_factor
):
    """
    Aggregate the damped MV forecasts of all players
    for every manager.

    Players without an ML prediction use their current
    Kickbase 24h market value trend as a fallback.
    """

    predictions = player_forecast_df.copy()
    predictions["player_id"] = predictions["player_id"].astype(str)

    total_prediction_lookup = dict(
        zip(
            predictions["player_id"],
            predictions["predicted_mv_change_until_matchday"]
        )
    )

    next_update_lookup = dict(
        zip(
            predictions["player_id"],
            predictions["predicted_next_update"]
        )
    )

    last_update_lookup = dict(
        zip(
            predictions["player_id"],
            predictions["predicted_last_update"]
        )
    )

    budget_df = manager_budgets_df.copy()
    budget_df["User_Key"] = (
        budget_df["User"]
        .astype(str)
        .str.strip()
    )

    managers = get_managers(
        token,
        league_id
    )

    results = []

    for manager_name, manager_id in managers:

        squad_data = get_manager_squad(
            token,
            league_id,
            manager_id
        )

        squad_players = squad_data.get("it", [])

        lineup_by_lo = [
            player
            for player in squad_players
            if player.get("lo", 0) > 0
        ]

        print(
            [
                (
                    player.get("pn"),
                    player.get("lo"),
                    player.get("pos")
                )
                for player in lineup_by_lo
            ]
        )

        last_24h_change = 0.0
        predicted_next_update = 0.0
        predicted_last_update = 0.0
        predicted_total_change = 0.0
        
        starting_lineup_value_at_md = 0.0
        bench_value_at_md = 0.0
        starting_players = 0
        bench_players = 0
        
        ml_players = 0
        fallback_players = 0
        missing_players = 0

        for player in squad_players:

            player_id = str(
                player.get("pi")
            )

            player_current_mv = float(
                player.get("mv") or 0
            )
            
            player_predicted_change = 0.0

            # Current Kickbase trend.
            # This also works for players missing from the ML dataset.
            current_24h_change = player.get(
                "tfhmvt"
            )

            if (
                current_24h_change is not None
                and pd.notna(current_24h_change)
            ):
                last_24h_change += float(
                    current_24h_change
                )

            # -----------------------------
            # Normal ML prediction
            # -----------------------------
            if player_id in total_prediction_lookup:

                total_change = (
                    total_prediction_lookup.get(
                        player_id,
                        0
                    )
                )

                if pd.notna(total_change):
                    player_predicted_change = float(
                        total_change
                    )

                next_change = (
                    next_update_lookup.get(
                        player_id,
                        0
                    )
                )

                final_change = (
                    last_update_lookup.get(
                        player_id,
                        0
                    )
                )

                if pd.notna(total_change):
                    predicted_total_change += float(
                        total_change
                    )

                if pd.notna(next_change):
                    predicted_next_update += float(
                        next_change
                    )

                if pd.notna(final_change):
                    predicted_last_update += float(
                        final_change
                    )

                ml_players += 1

                player_value_at_md = (
                    player_current_mv
                    + player_predicted_change
                )
                
                # lo = 0 is the goalkeeper.
                # lo = 1..10 are starting outfield players.
                # lo = None means the player is currently not in the lineup.
                if player.get("lo") is None:
                
                    bench_value_at_md += (
                        player_value_at_md
                    )
                
                    bench_players += 1
                
                else:
                
                    starting_lineup_value_at_md += (
                        player_value_at_md
                    )
                
                    starting_players += 1
                
                continue

                        # -----------------------------
            # Fallback prediction
            # -----------------------------
            if (
                current_24h_change is not None
                and pd.notna(current_24h_change)
            ):

                current_24h_change = float(
                    current_24h_change
                )

                # Without an ML prediction, use the sign of the
                # current Kickbase 24h trend to choose the decay.
                if current_24h_change >= 0:
                    fallback_decay = (
                        positive_decay_factor
                    )
                else:
                    fallback_decay = (
                        negative_decay_factor
                    )

                # The known value is today's change.
                # Therefore the first FUTURE update is already
                # one decay step after today's momentum.
                fallback_next_change = (
                    current_24h_change
                    * fallback_decay
                )

                if market_updates_until_matchday <= 0:

                    fallback_total_change = 0.0
                    fallback_last_change = 0.0

                else:

                    if fallback_decay == 1:

                        fallback_cumulative_multiplier = float(
                            market_updates_until_matchday
                        )

                    else:

                        fallback_cumulative_multiplier = (
                            1
                            - fallback_decay
                            ** market_updates_until_matchday
                        ) / (
                            1
                            - fallback_decay
                        )

                    fallback_last_multiplier = (
                        fallback_decay
                        ** (
                            market_updates_until_matchday
                            - 1
                        )
                    )

                    fallback_last_change = (
                        fallback_next_change
                        * fallback_last_multiplier
                    )

                    fallback_total_change = (
                        fallback_next_change
                        * fallback_cumulative_multiplier
                    )

                predicted_next_update += (
                    fallback_next_change
                )

                predicted_last_update += (
                    fallback_last_change
                )

                predicted_total_change += (
                    fallback_total_change
                )

                player_predicted_change = (
                    fallback_total_change
                )
                
                player_value_at_md = (
                    player_current_mv
                    + player_predicted_change
                )
                
                if player.get("lo") is None:
                
                    bench_value_at_md += (
                        player_value_at_md
                    )
                
                    bench_players += 1
                
                else:
                
                    starting_lineup_value_at_md += (
                        player_value_at_md
                    )
                
                    starting_players += 1
                
                fallback_players += 1

            else:
                missing_players += 1

        manager_key = str(
            manager_name
        ).strip()

        manager_budget_row = budget_df[
            budget_df["User_Key"]
            == manager_key
        ]

        if manager_budget_row.empty:
            print(
                f"Warning: No budget information "
                f"found for {manager_name}"
            )
            continue

        current_team_value = float(
            manager_budget_row.iloc[0][
                "Team Value"
            ]
        )

        current_budget = float(
            manager_budget_row.iloc[0][
                "Budget"
            ]
        )

        predicted_team_value = (
            current_team_value
            + predicted_total_change
        )

        predicted_budget = (
            current_budget
            + future_login_bonus
        )

        cash_after_bench_sales = (
            predicted_budget
            + bench_value_at_md
        )

        manager_value = (
            predicted_team_value
            + predicted_budget
        )

        squad_size = len(
            squad_players
        )

        total_predicted_players = (
            ml_players
            + fallback_players
        )

        coverage = (
            f"{total_predicted_players}/{squad_size}"
            if squad_size > 0
            else "0/0"
        )

        results.append({
            "User": manager_name,
            "Team Value Now": current_team_value,
            "Last 24h MV Change": last_24h_change,
            "Pred. Next Update": predicted_next_update,
            "Pred. Last Update": predicted_last_update,
            "Pred. MV Change to MD": predicted_total_change,
            "Team Value @ MD": predicted_team_value,
        
            "Starting XI @ MD": starting_lineup_value_at_md,
            "Bench @ MD": bench_value_at_md,
        
            "Cash Now": current_budget,
            "Future Login Bonus": future_login_bonus,
            "Cash @ MD": predicted_budget,
            "Cash after Bench Sales @ MD": cash_after_bench_sales,
        
            "Manager Value @ MD": manager_value,
        
            "Starting Players": starting_players,
            "Bench Players": bench_players,
            "ML Players": ml_players,
            "Fallback Players": fallback_players,
            "Prediction Coverage": coverage
        })

    result_df = pd.DataFrame(
        results
    )

    if not result_df.empty:
        result_df = result_df.sort_values(
            "Manager Value @ MD",
            ascending=False,
            ignore_index=True
        )

    return result_df
