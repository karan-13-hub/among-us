#!/usr/bin/env python3
# usage: python app.py

import os
import sys
import asyncio
import uuid
import threading
import random
import json
import time
import io
import contextlib
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_socketio import SocketIO, emit

sys.path.append(os.path.join(os.path.abspath(".."), "among-agents"))
sys.path.append(os.path.abspath(".."))

from amongagents.envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME, THREE_MEMBER_GAME
from amongagents.envs.game import AmongUs
from dotenv import load_dotenv

from utils import setup_experiment
from config import CONFIG

ROOT_PATH = os.path.abspath(".")
LOGS_PATH = os.path.join(ROOT_PATH, CONFIG["logs_path"])

load_dotenv()

experiment_name = CONFIG["experiment_name"]

BIG_LIST_OF_MODELS = [
    "microsoft/phi-4",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet:thinking",
    "openai/o3-mini-high",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-r1-distill-llama-70b",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-7b-instruct",
    "deepseek/deepseek-r1",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
]

TESTING_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
]

GAME_ARGS = {
    "game_config": FIVE_MEMBER_GAME,
    "include_human": True,  # Set to True for human players
    "test": False,
    "personality": False,
    "agent_config": {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        "IMPOSTOR_LLM_CHOICES": TESTING_MODELS,
        "CREWMATE_LLM_CHOICES": TESTING_MODELS,
    },
    "UI": False,
    "tournament_style": "random",  # Default tournament style
}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management
socketio = SocketIO(app)

# Store active games in memory
active_games = {}

# Context manager to suppress stderr
@contextlib.contextmanager
def suppress_stderr():
    stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = stderr

def setup_experiment_once():
    """Setup experiment directory and files"""
    # Check if experiment directory already exists
    experiment_dir = os.path.join(LOGS_PATH, experiment_name)
    if os.path.exists(experiment_dir):
        return
    
    # Only run setup if not already done
    if not os.environ.get("EXPERIMENT_PATH"):
        os.environ["EXPERIMENT_PATH"] = experiment_dir
        print(f"Setting up experiment {experiment_name}")
        setup_experiment(experiment_name, LOGS_PATH, CONFIG["date"], CONFIG["commit_hash"], GAME_ARGS)

def get_next_game_index():
    """Get the next game index by reading the experiment log file"""
    # Check if EXPERIMENT_PATH is set
    if "EXPERIMENT_PATH" not in os.environ:
        # If not set, return default index of 1
        return 1
        
    experiment_file_path = os.path.join(os.environ["EXPERIMENT_PATH"], "experiment-details.txt")
    
    # Default to 1 if file doesn't exist or no games found
    next_index = 1
    
    if os.path.exists(experiment_file_path):
        with open(experiment_file_path, 'r') as file:
            content = file.read()
            # Look for "Game X started" entries
            import re
            game_entries = re.findall(r'Game (\d+) started', content)
            if game_entries:
                # Get the highest game index and increment by 1
                next_index = max(map(int, game_entries)) + 1
    
    return next_index

async def run_game_instance(session_id):
    """Run a single game instance for a specific session"""
    # Ensure setup has been done before starting a new game
    if not os.environ.get("EXPERIMENT_PATH"):
        setup_experiment_once()
        
    # Get the next game index
    game_index = get_next_game_index()
    
    # Append game index to the experiment details
    with open(os.path.join(os.environ["EXPERIMENT_PATH"], "experiment-details.txt"), "a") as experiment_file:
        experiment_file.write(f"\nGame {game_index} started.\n")
    
    # Create a new game instance
    game = AmongUs(
        game_config=GAME_ARGS["game_config"],
        include_human=GAME_ARGS["include_human"],
        test=GAME_ARGS["test"],
        personality=GAME_ARGS["personality"],
        agent_config=GAME_ARGS["agent_config"],
        UI=None,  # No UI, using Flask instead
        game_index=game_index,
    )
    
    # Initialize the game
    game.initialize_game()
    
    # Store the game instance
    active_games[session_id] = game
    
    # Start the game in a separate thread with stderr suppression
    def run_game_with_suppression():
        with suppress_stderr():
            asyncio.run(game.run_game())
            
    threading.Thread(target=run_game_with_suppression, daemon=True).start()
    
    return game

def get_game_state(game):
    """Get the current game state"""
    if not game:
        return None
    
    # Get player information
    players = []
    for player in game.players:
        player_info = {
            "name": player.name,
            "color": player.color,
            "identity": player.identity,
            "location": player.location,
            "is_alive": player.is_alive,
            "tasks": [task.name for task in player.tasks] if hasattr(player, 'tasks') else []
        }
        players.append(player_info)
    
    # Get game phase information
    phase_info = {
        "current_phase": game.current_phase,
        "discussion_rounds_left": game.discussion_rounds_left if hasattr(game, 'discussion_rounds_left') else None
    }
    
    # Get activity log
    activity_log = game.activity_log[-20:] if hasattr(game, 'activity_log') else []
    
    # Get important activity log
    important_activity_log = game.important_activity_log if hasattr(game, 'important_activity_log') else []
    
    # Get camera record
    camera_record = game.camera_record if hasattr(game, 'camera_record') else {}
    
    # Get votes
    votes = game.votes if hasattr(game, 'votes') else {}
    
    # Get vote info
    vote_info = game.vote_info_one_round if hasattr(game, 'vote_info_one_round') else {}
    
    # Get task completion
    task_completion = game.task_assignment.check_task_completion() if hasattr(game, 'task_assignment') else 0
    
    # Get game over status
    game_over = game.check_game_over() if hasattr(game, 'check_game_over') else 0
    
    # Get winner
    winner = None
    if game_over > 0:
        winner_map = {
            1: "Impostors win! (Crewmates being outnumbered or tied to impostors)",
            2: "Crewmates win! (Impostors eliminated)",
            3: "Crewmates win! (All task completed)",
            4: "Impostors win! (Time limit reached)"
        }
        winner = winner_map.get(game_over, "Unknown")
    
    return {
        "game_index": game.game_index,
        "timestep": game.timestep,
        "players": players,
        "phase_info": phase_info,
        "activity_log": activity_log,
        "important_activity_log": important_activity_log,
        "camera_record": camera_record,
        "votes": votes,
        "vote_info": vote_info,
        "task_completion": task_completion,
        "game_over": game_over,
        "winner": winner,
        "last_update": time.time()
    }

@app.route('/')
def index():
    # Create new session if none exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html')

@socketio.on('start_game')
def on_start_game(data):
    player_name = data.get('playerName')
    
    if not player_name:
        emit('error', {'message': 'Player name is required'})
        return
    
    # Create a new session ID if needed
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    # Start a new game
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    game = loop.run_until_complete(run_game_instance(session_id))
    
    # Store player information
    player_id = request.sid
    game.add_player(player_id, player_name)
    
    # Send initial game state
    game_state = get_game_state(game)
    emit('game_state', game_state)
    
    # Send game log update
    emit('game_log', {
        'message': f"Game started with player {player_name}",
        'timestamp': time.time()
    })
    
    # Start a background thread to send game updates
    def send_game_updates():
        while True:
            if session_id in active_games:
                game = active_games[session_id]
                game_state = get_game_state(game)
                
                # Send game state update
                socketio.emit('game_state', game_state, room=request.sid)
                
                # Send activity log updates
                if game.activity_log:
                    latest_activity = game.activity_log[-1]
                    socketio.emit('game_log', {
                        'message': latest_activity,
                        'timestamp': time.time()
                    }, room=request.sid)
                
                # Check if game is over
                if game.check_game_over() > 0:
                    winner = None
                    game_over = game.check_game_over()
                    winner_map = {
                        1: "Impostors win! (Crewmates being outnumbered or tied to impostors)",
                        2: "Crewmates win! (Impostors eliminated)",
                        3: "Crewmates win! (All task completed)",
                        4: "Impostors win! (Time limit reached)"
                    }
                    winner = winner_map.get(game_over, "Unknown")
                    
                    socketio.emit('game_log', {
                        'message': f"Game Over! {winner}",
                        'timestamp': time.time()
                    }, room=request.sid)
                    
                    # Remove the game from active games
                    del active_games[session_id]
                    break
            
            time.sleep(1)  # Update every second
    
    threading.Thread(target=send_game_updates, daemon=True).start()

@socketio.on('disconnect')
def on_disconnect():
    player_id = request.sid
    
    # Find the game this player was in
    for session_id, game in active_games.items():
        if session_id == session.get('session_id'):
            # Remove the player from the game
            game.remove_player(player_id)
            
            # If no players left, clean up the game
            if not game.players:
                del active_games[session_id]
            
            break

if __name__ == "__main__":
    # Setup experiment
    setup_experiment_once()
    
    # Run the Flask app
    socketio.run(app, debug=True, host='0.0.0.0', port=3000) 