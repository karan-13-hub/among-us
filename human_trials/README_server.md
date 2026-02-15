# AmongUs Game Server

This server provides an API to create and manage AmongUs games.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements_server.txt
   ```

2. Run the server:
   ```
   python server.py
   ```

The server will start on `XXXX`.

## API Endpoints

### Start a New Game

```
POST /api/start_game
```

Creates a new AmongUs game and returns the game configuration.

**Request Body (JSON):**
```json
{
    "game_config": "FIVE_MEMBER_GAME",  // Optional, defaults to FIVE_MEMBER_GAME
    "include_human": true,  // Optional, defaults to True
    "tournament_style": "random",  // Optional, defaults to "random"
    "impostor_model": "meta-llama/llama-3.3-70b-instruct",  // Optional
    "crewmate_model": "meta-llama/llama-3.3-70b-instruct"  // Optional
}
```

**Response:**
```json
{
    "game_id": 1,
    "status": "created",
    "config": {
        "game_config": "FIVE_MEMBER_GAME",
        "include_human": true,
        "tournament_style": "random",
        "impostor_model": "meta-llama/llama-3.3-70b-instruct",
        "crewmate_model": "meta-llama/llama-3.3-70b-instruct"
    }
}
```

### Get Game Information

```
GET /api/game/{game_id}
```

Returns information about a specific game.

**Response:**
```json
{
    "game_id": 1,
    "status": "created",
    "players": [...],
    "current_phase": "unknown",
    "summary": {}
}
```

### Get All Games

```
GET /api/games
```

Returns information about all active games.

**Response:**
```json
{
    "games": [
        {
            "game_id": 1,
            "status": "created",
            "current_phase": "unknown"
        },
        {
            "game_id": 2,
            "status": "running",
            "current_phase": "discussion"
        }
    ]
}
```

### Run a Game

```
POST /api/run_game/{game_id}
```

Runs a specific game and returns the results.

**Response:**
```json
{
    "game_id": 1,
    "status": "completed",
    "results": {
        // Game summary and results
    }
}
```

## Example Usage

### Using cURL

1. Start a new game:
   ```
   curl -X POST XXXX \
     -H "Content-Type: application/json" \
     -d '{"game_config": "FIVE_MEMBER_GAME", "include_human": true}'
   ```

2. Run the game:
   ```
   curl -X POST XXXX
   ```

3. Get game information:
   ```
   curl XXXX
   ```

### Using Python

```python
import requests
import json

# Start a new game
response = requests.post(
    "XXXX",
    json={"game_config": "FIVE_MEMBER_GAME", "include_human": True}
)
game_data = response.json()
game_id = game_data["game_id"]

# Run the game
response = requests.post(f"XXXX{game_id}")
results = response.json()

# Get game information
response = requests.get(f"XXXX{game_id}")
game_info = response.json() 