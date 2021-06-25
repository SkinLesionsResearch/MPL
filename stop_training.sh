pid=`ps -aux | grep main.py | awk '{print $2}' | head -n 1` 
echo "kill process with $pid"
ps -aux | grep main.py | awk '{print $2}' | head -n 1 | xargs kill $pid
