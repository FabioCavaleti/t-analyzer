uvicorn python_backend.server:app --host 0.0.0.0 --port 8000 --reload > uvicorn.log 2>&1 &

./cpp_modules/build/src/bt-analyzer > /dev/null 2>&1 &