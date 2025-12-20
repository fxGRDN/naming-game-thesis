nohup uv run $1 >/dev/null 2>&1 &
pid=$!
disown
echo "$pid" > "background_$1.pid"