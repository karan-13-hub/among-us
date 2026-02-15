python main.py --name "phi_llama_100_games_v3" --num_games 100 --crewmate_llm "microsoft/phi-4" --impostor_llm "meta-llama/llama-3.3-70b-instruct"
python main.py --name "phi_phi_100_games_v3" --num_games 100 --crewmate_llm "microsoft/phi-4" --impostor_llm "microsoft/phi-4"
python main.py --name "llama_llama_100_games_v3" --num_games 100 --crewmate_llm "meta-llama/llama-3.3-70b-instruct" --impostor_llm "meta-llama/llama-3.3-70b-instruct"
python main.py --name "llama_phi_100_games_v3" --num_games 100 --crewmate_llm "meta-llama/llama-3.3-70b-instruct" --impostor_llm "microsoft/phi-4"