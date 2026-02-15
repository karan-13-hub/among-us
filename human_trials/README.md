# Among Us: A Sandbox for Agentic Deception

This project implements a web-based version of the popular game "Among Us" as a sandbox for studying agentic deception in human-AI interactions.

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
  - `index.html`: Main game interface
- `static/`: Static assets
  - `map.png`: Game map image

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open a web browser and navigate to `XXXX`

## Game Rules

- Players are randomly assigned roles as either Crewmates or Impostors
- Crewmates must complete tasks while avoiding the Impostor
- Impostor must eliminate Crewmates without being discovered
- Players can report dead bodies and call emergency meetings
- During meetings, players vote to eject someone they suspect is the Impostor

## Development

This project is built with:
- Flask: Web framework
- Flask-SocketIO: Real-time communication
- Bootstrap: UI styling
- JavaScript: Client-side game logic 