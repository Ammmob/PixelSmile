import torch

def lora_processors(model):
    """Extract LoRA processors from model"""
    processors = {}
    
    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
        return processors
    
    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)
    
    return processors