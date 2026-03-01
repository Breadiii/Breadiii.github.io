# Local ran - Shelter pressure evaluation

## Pressure-Location calculator
how to locally host  
Run the app.py python page  
In a new terminal run:  
python -m pip install --user -r backend\requirements.txt  
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000  
 
then in a new terminal  
python -m http.server 8080 --bind 127.0.0.1 --directory frontend  

then open frontend on http://127.0.0.1:8080 to access the calcuclator  

## The heatmap
Run the heatmap_app.py  
Then run this is the termrinal  
python -m uvicorn backend.heatmap_app:app --host 127.0.0.1 --port 8001  
open http://127.0.0.1:8080/Headmap.html to access the heatmap  


## SDSS 2026 Datathon main repository
main project: https://github.com/Ancientrains/sdss-Datathon-team44  

## SDSS 2026 Datathon Devpost submission page
https://devpost.com/software/public_services-data-file
https://sdss-datathon-2026.devpost.com 
