nohup uv run $1.py >"${1}.log" 2>&1 &
pid=$!
disown
echo "$pid" > "background_$1.pid"