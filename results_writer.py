import pandas as pd
import yaml
from pathlib import Path


def create_setting_folder(setting, number):
    path ="results/"+setting.params["filename"]+"/"
    path = Path(path+"setting_"+str(number))
    path.mkdir(parents=True, exist_ok=True)
    param_path = path / "setting.yml"
    with open(param_path, 'w') as yaml_file:
        yaml.dump(setting.params, yaml_file)
        

def write_scores(scores, setting, number, seed):
    
    df = pd.DataFrame(scores, index=[seed])
    
    path ="results/"+setting.params["filename"]+"/"
    path = Path(path+"setting_"+str(number)+"/"+"setting_"+str(number)+".csv")
    
    if path.is_file():
        df.to_csv(path, mode='a', header=False)
    else:
        df.to_csv(path, mode='a', header=True)
        
        
def write_threshold_scores(scores, setting, number, seed):
    df = pd.DataFrame(scores, index=[seed])
    
    path ="results/"+setting.params["filename"]+"/"
    
    path = Path(path+"setting_"+str(number)+"/"+"setting_"+str(number)+"_thresholds.csv")
    
    if path.is_file():
        df.to_csv(path, mode='a', header=False)
    else:
        df.to_csv(path, mode='a', header=True)
    
    
   