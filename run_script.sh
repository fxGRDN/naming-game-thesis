SCRIPT_NAME=$(basename "$1")
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

nohup uv run $@ >"logs/${SCRIPT_NAME}-${TIMESTAMP}.log" 2>&1 &
pid=$!
disown
echo "$pid" > "logs/background_${SCRIPT_NAME}-${TIMESTAMP}.pid"