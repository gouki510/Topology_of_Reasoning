def model_name_mapping(model_name_or_path):
    if 'llama-2' in model_name_or_path:
        if '13b' in model_name_or_path:
            return "meta-llama/Llama-2-13b-hf"
        else:
            return "meta-llama/Llama-2-7b-hf"
    else:
        return model_name_or_path