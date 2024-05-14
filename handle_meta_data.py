import yaml
import shutil
from itertools import product
from pathlib import Path
from collections import namedtuple



def get_state(exp_name):
      
    result_path ="results/"+exp_name+"/"
    paths = Path(result_path).glob("*")
    settings = filter(lambda p : str(p.name).startswith("setting"),paths)
    indices = map(lambda p : int(str(p.name).split('_')[1]), settings)
    return max(indices)

def built_config_with_default(wandb_config, dataset):
    config_dict = dict(wandb_config.items())

    path = Path("default_parameter/default_parameter_" + dataset+".yml")
    config_default = load_yml(path)
    config_default.update(config_dict)

    config = namedtuple('Config', config_default.keys())(**config_default)

    return config
    

def delete_last_setting(args):
    path = args.experiment[0]
    exp_name = Path(path).stem
    
    setting_number = get_state(exp_name)
    shutil.rmtree("results/"+exp_name+"/setting_"+str(setting_number))
    return setting_number

def find_pars_yaml(args):
    
    paths = []
    ret = []
    #only path
    if args.experiment != None:
        
        paths = []
        
        if isinstance(args.experiment, list):
            for p in args.experiment:
                paths.append(Path(p))
                
        else:
            path = Path(args.experiment)
            paths.append(path)
            
        return paths
       
            
    #all yaml
    else:
        paths = Path("experiments/").glob("*.yml")
    
    for p in paths:
        yml = yaml.safe_load(p.open())
        if yml.get("alldone"):
            continue
        else:
            ret.append(p)
    return ret

def load_yml(path):
    yml_dict = yaml.safe_load(path.open())
    filename = path.stem
    yml_dict["filename"] = [filename]
    return yml_dict

def get_combinations_of_yml(yml_dict : dict):
    dictlist =yml_dict.items()
    dictlist_key_value = [ [(x,z) for z in y] for (x, y) in dictlist]
    all_key_value_combin = list(product(*dictlist_key_value))
    return all_key_value_combin
    

def fill_defaults(setting, path=Path("default_parameters/default_parameters.yml")):
    default_dict = load_yml(path)
    all_params = dict(setting)
    default_dict.update(all_params)
    return default_dict