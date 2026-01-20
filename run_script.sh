nohup uv run $1.py >"${1}-$(date +%Y%m%d%H%M%S).log" 2>&1 &
pid=$!
disown
echo "$pid" > "background_$1-$(date +%Y%m%d%H%M%S).pid"