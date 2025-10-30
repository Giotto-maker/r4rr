import os
import importlib


def get_all_models():
    models_dir = os.path.join(os.path.dirname(__file__), "")
    return [
        model.split(".")[0]
        for model in os.listdir(models_dir)
        if not model.find("__") > -1 and "py" in model
    ]


names = {}
for model in get_all_models():
    mod = importlib.import_module("models." + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace("_", "")]
    names[model] = getattr(mod, class_name)



def get_model(args, encoder, decoder, n_images, c_split):
    print("Available models:", list(names.keys()))
    if args.model == "cext":
        return names[args.model](encoder, n_images=n_images, c_split=c_split)
    elif args.model in [
        "mnistdpl",
        "mnistsl",
        "shieldedmnist",            # MNIST PNets
        "proshieldedmnist",
        "promnistsl",               
        "promnistltn",      
        "promnistdpl",      
        "promnmathdpl",     
        "promnmathcbm",     
        "mnistltn",
        "kanddpl",
        "kandltn",
        "kandpreprocess",
        "kandclip",
        "kandsl",           
        "kandsloneembedding",       # TODO: remove
        "prokandsloneembedding",    # TODO: remove
        "prokanddpl",               # KAND PNets
        "prokandsl",                
        "prokandltn",               
        "kanddplsinglejoint",       # KAND enhanced baseline 
        "kanddplsingledisj",        
        "kandslsinglejoint",
        "kandslsingledisj",
        "kandltnsinglejoint",
        "kandltnsingledisj",
        "minikanddpl",
        "mnistpcbmdpl",
        "mnistpcbmsl",
        "mnistpcbmltn",
        "mnistclip",
        "sddoiadpl",
        "sddoiacbm",
        "sddoialtn",
        "presddoiadpl",
        "boiadpl",
        "probddoiadpl",
        "bddoiadpldisj",
        "mnistcbm",
        "boiacbm",
        "boialtn",
        "kandcbm",
        "mnistnn",
        "kandnn",
        "sddoiann",
        "sddoiaclip",
        "boiann",
        "xorcbm",
        "xornn",
        "xordpl",
        "mnmathnn",
        "mnmathcbm",
        "mnmathdpl"
    ]:
        return names[args.model](
            encoder, n_images=n_images, c_split=c_split, args=args
        )  # only discriminative
    else:
        return names[args.model](
            encoder, decoder, n_images=n_images, c_split=c_split, args=args
        )
