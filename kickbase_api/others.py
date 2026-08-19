from kickbase_api.config import BASE_URL, get_json_with_token
from datetime import datetime
import requests

# All other functions that don't fit anywhere else

def get_all_teams(token, competition_id):
    """Get all teams in a competition."""

    url = f"{BASE_URL}/competitions/{competition_id}/table"
    data = get_json_with_token(url, token)

    teams = [
        {
            "team_id": item.get("tid"),   # Team-ID
            "team_name": item.get("tn")   # Team Name
        }
        for item in data.get("it", [])
    ]

    return teams

def get_matchdays(token, competition_id):
    """Get all matchdays in a competition with the latest date for each matchday."""

    url = f"{BASE_URL}/competitions/{competition_id}/matchdays"
    data = get_json_with_token(url, token)

    matches = [
        {
            "day": match.get("day"),
            "date": match.get("dt")
        }
        for item in data.get("it", [])
        for match in item.get("it", [])
    ]

    max_dates_per_day = {}
    for m in matches:
        day = m["day"]
        date = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))  # ISO -> datetime
        if day not in max_dates_per_day or date > max_dates_per_day[day]:
            max_dates_per_day[day] = date

    result = [{"day": day, "date": max_dates_per_day[day].isoformat()} for day in sorted(max_dates_per_day)]

    return result

def get_achievement_reward(token, league_id, achievement_id):
    """Get the reward and how often this was achieved by the user for a specific achievement in a league."""

    url = f"{BASE_URL}/leagues/{league_id}/user/achievements/{achievement_id}"
    data = get_json_with_token(url, token)

    amount = data.get("ac", 0)
    reward = data.get("er", 0)
    
    return amount, reward
