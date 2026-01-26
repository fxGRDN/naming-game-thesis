SCRIPT_DIR=$(dirname "$1")
SCRIPT_NAME=$(basename "$1" .py)
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

nohup uv run "$1.py" >"logs/${SCRIPT_NAME}-${TIMESTAMP}.log" 2>&1 &
pid=$!
disown
echo "$pid" > "logs/background_${SCRIPT_NAME}-${TIMESTAMP}.pid"