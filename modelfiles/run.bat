ollama create -f .\gpt-oss-claude-20b-modelfile.txt gpt-oss-claude
ollama create -f .\gpt-oss-20b-modelfile.txt gpt-oss
ollama create -f .\llama3.2-modelfile.txt llama3.2
ollama create -f .\llava-modelfile.txt llava
ollama create -f .\qwen2.5-coder-modelfile.txt qwen2.5-coder

ollama run gpt-oss-claude /bye
ollama run gpt-oss /bye
ollama run llama3.2 /bye
ollama run llava /bye
ollama run qwen2.5-coder /bye

ollama pull mxbai-embed-large