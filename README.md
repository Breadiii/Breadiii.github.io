# Breadiii.github.io  

In terminal  
python -m pip install --user -r backend\requirements.txt  
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000  

then in a new terminal  
python -m http.server 8080 --bind 127.0.0.1 --directory frontend  

then open frontend on http://127.0.0.1:8080  
