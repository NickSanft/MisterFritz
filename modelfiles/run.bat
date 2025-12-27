ollama create -f .\gpt-oss-20b-modelfile.txt gpt-oss
ollama create -f .\llama3.2-modelfile.txt llama3.2

ollama run gpt-oss /bye
ollama run llama3.2 /bye

ollama pull mxbai-embed-large