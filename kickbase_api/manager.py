from kickbase_api.config import BASE_URL, get_json_with_token

# All functions related to manager data

def get_managers(token, league_id):
    """Get a list of all managers in the league with their IDs and names."""

    url = f"{BASE_URL}/leagues/{league_id}/ranking"
    data = get_json_with_token(url, token)

    user_info = [(user["n"], user["i"]) for user in data["us"]]

    return user_info

def get_manager_info(token, league_id, manager_id):
    """Get detailed information about a specific manager in the league."""

    url = f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/dashboard"
    data = get_json_with_token(url, token)

    return data

def get_manager_performance(token, league_id, manager_id, manager_name):
    """Get performance data for a specific manager in the league."""

    url = f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/performance"
    data = get_json_with_token(url, token)

    # Current Kickbase season 2026/2027
    current_season_id = "42"

    tp_value = 0

    for season in data.get("it", []):
        if season.get("sid") == current_season_id:
            tp_value = season.get("tp", 0)
            break
    else:
        print(f"Warning: Season ID '{current_season_id}' not found for {manager_name}, using 0")
    

    return {
        "name": manager_name,
        "tp": tp_value
    }

def get_manager_squad(token, league_id, manager_id):
    """Get the current squad of a manager."""
    url = f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/squad"
    data = get_json_with_token(url, token)

    return data
