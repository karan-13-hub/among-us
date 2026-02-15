// Socket.io connection
const socket = io();

// Game state
let gameState = {
    players: [],
    currentPlayer: null,
    tasks: [],
    isImpostor: false,
    isDead: false,
    meetingInProgress: false
};

// DOM Elements
const loginScreen = document.getElementById('login-screen');
const gameScreen = document.getElementById('game-screen');
const playerNameInput = document.getElementById('player-name');
const startButton = document.getElementById('start-game');
const gameLog = document.getElementById('game-log');

// Start game
startButton.addEventListener('click', () => {
    const playerName = playerNameInput.value.trim();
    
    if (playerName) {
        socket.emit('start_game', { playerName });
        loginScreen.classList.add('d-none');
        gameScreen.classList.remove('d-none');
        
        // Add initial log entry
        addLogEntry('System', 'Connecting to game server...', 'info');
    }
});

// Socket event handlers
socket.on('game_state', (state) => {
    gameState = state;
    updateGameUI();
});

socket.on('game_log', (data) => {
    addLogEntry('Game', data.message, 'info');
});

socket.on('error', (data) => {
    addLogEntry('Error', data.message, 'error');
});

// UI update functions
function updateGameUI() {
    // Update game log with game state information
    if (gameState.phase_info) {
        addLogEntry('Phase', `Current phase: ${gameState.phase_info.current_phase}`, 'info');
    }
    
    if (gameState.players && gameState.players.length > 0) {
        addLogEntry('Players', `Players in game: ${gameState.players.map(p => p.name).join(', ')}`, 'info');
    }
    
    if (gameState.activity_log && gameState.activity_log.length > 0) {
        const latestActivity = gameState.activity_log[gameState.activity_log.length - 1];
        addLogEntry('Activity', latestActivity, 'info');
    }
    
    if (gameState.game_over) {
        addLogEntry('Game Over', gameState.winner || 'Game has ended', 'success');
    }
}

function addLogEntry(source, message, type = 'info') {
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${type}`;
    
    const timestamp = document.createElement('span');
    timestamp.className = 'log-timestamp';
    timestamp.textContent = `[${new Date().toLocaleTimeString()}]`;
    
    const sourceSpan = document.createElement('span');
    sourceSpan.className = 'log-source';
    sourceSpan.textContent = `[${source}] `;
    
    const messageSpan = document.createElement('span');
    messageSpan.className = 'log-message';
    messageSpan.textContent = message;
    
    logEntry.appendChild(timestamp);
    logEntry.appendChild(sourceSpan);
    logEntry.appendChild(messageSpan);
    
    gameLog.appendChild(logEntry);
    
    // Auto-scroll to the bottom
    gameLog.scrollTop = gameLog.scrollHeight;
} 