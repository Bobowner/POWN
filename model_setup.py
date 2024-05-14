from models.OpenGraphCon import OpenGraph
from models.OpenGCN import OpenGCN
from models.OpenSSL import OpenSSL
from models.OpenSpectral import OpenSpectral
from models.DGI import DeepGraphInfomax, corrupt, readout
from torch_geometric.nn.models import GCN


def setup_model(data, device, config, num_protos, log_all):
    if config.model == "pown":
        model = setup_open_graph_model(data, device, config, num_protos, log_all)
    elif config.model == "gcn":
        model = setup_open_GCN(data, device, config, log_all)
    elif config.model == "openssl":
        model = setup_open_DGI(data, device, config, log_all)
    elif config.model == "spectral":
        model = setup_open_spectral(data, device, config, log_all)
    else:
        raise NotImplementedError("Your choosen model: " + config.model + " is not defined.")

    return model


def setup_open_spectral(data, device, config, log_all):

    model = OpenSpectral(out_dim = config.num_protos, 
                    known_classes = data.known_classes, 
                    unknown_classes = data.unknown_classes,
                    device = device, 
                    log_all = log_all)

    return model

def setup_open_GCN(data, device, config, log_all):
    
    hidden_channels = config.hidden_channels
    dropout = config.dropout
    
    model =  OpenGCN(in_channels = data.x.shape[1], 
                      hidden_channels = hidden_channels,
                      out_dim = config.num_protos,
                      dropout = config.dropout, 
                      num_layers = config.num_layers, 
                      known_classes = data.known_classes, 
                      unknown_classes = data.unknown_classes, 
                      device = device,
                      log_all = log_all)
    return model


def setup_open_DGI(data, device, config, log_all):

    hidden_channels = config.hidden_channels
    num_layers = config.num_layers
    dropout = config.dropout
    
    encoder = GCN(in_channels = data.x.shape[1],
                  hidden_channels = config.hidden_channels,
                  out_channels = None,
                  dropout = config.dropout,
                  num_layers =  config.num_layers)

    dgi = DeepGraphInfomax(hidden_channels = config.hidden_channels, 
                           encoder = encoder,
                           summary = readout,
                           corruption = corrupt)

    
    og_model = OpenSSL(encoder=encoder, 
                       ssl=dgi, 
                       out_dim = config.num_protos, 
                       log_all=log_all, 
                       device=device)
    return og_model



def setup_open_graph_model(data, device, config, num_protos, log_all):
    
    hidden_channels = config.hidden_channels
    num_layers = config.num_layers
    dropout = config.dropout
    
    encoder = GCN(in_channels = data.x.shape[1],
                  hidden_channels = config.hidden_channels,
                  out_channels = None,
                  dropout = config.dropout,
                  num_layers =  config.num_layers)
    
    
    
    dgi = DeepGraphInfomax(hidden_channels = config.hidden_channels, 
                           encoder = encoder,
                           summary = readout,
                           corruption = corrupt)

    
    
    og_model = OpenGraph(ssl = dgi, 
                         hidden_channels = config.hidden_channels, 
                         num_protos = num_protos,
                         known_classes = data.known_classes, 
                         unknown_classes = data.unknown_classes, 
                         device = device, 
                         sup_loss_weight = config.sup_loss_weight,
                         pseudo_loss_weight = config.pseudo_loss_weight,
                         unsup_loss_weight = config.unsup_loss_weight,
                         entropy_loss_weight = config.entropy_loss_weight,
                         geometric_loss_weight = config.geometric_loss_weight,
                         ood_percentile = config.ood_percentile, 
                         proto_type=config.proto_type,
                         lp_hop = config.lp_hop,
                         sup_temp = config.sup_temp,
                         pseudo_temp = float(config.pseudo_temp),
                         lp_min_seeds = float(config.lp_min_seeds), 
                         entropy_threshold = config.entropy_threshold,
                         log_all = log_all,
                         pseudo_label_method = config.pseudo_label_method)
    return og_model