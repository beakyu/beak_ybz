def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path
    if model_type == "gemini_only":  
        from models.gemini_only import GeminiOnlyMethod
        model = GeminiOnlyMethod()

    elif model_type == "internvl3_gemini":  
        from models.internvl3_gemini import InternVL3GeminiMethod  
        model = InternVL3GeminiMethod()  
 
    elif model_type == "uitars_qwen":  
        from models.uitars_qwen import UITarsQwenComboMethod  
        model = UITarsQwenComboMethod () 

    elif model_type == "qwen_uitars":  
        from models.qwen_uitars import QwenUITarsComboMethod 
        model = QwenUITarsComboMethod() 

    elif model_type == "uitars_qwen_uitars":  
        from models.qwen_uitars_uitars import UITarsQwenComboMethod 
        model = UITarsQwenComboMethod() 

    elif model_type == "uitars_internvl3":  
        from models.uitars_internvl3 import UITarsInternVL3ComboMethod
        model = UITarsInternVL3ComboMethod () 

    elif model_type == "internvl3_uitars":  
        from models.internvl3_uitars import InternVL3UITarsComboMethod
        model = InternVL3UITarsComboMethod () 

    elif model_type == "uitars_gemini":  
        from models.uitars_gemini import UITarsGeminiComboMethod
        model = UITarsGeminiComboMethod () 

    elif model_type == "gemini_uitars":  
        from models.gemini_uitars import GeminiUITarsComboMethod
        model = GeminiUITarsComboMethod ()

    elif model_type == "uitars_uitars":  
        from models.uitars_uitars import UITarsSelfRefineMethod
        model = UITarsSelfRefineMethod ()

    elif model_type == "qwen_uitars_uitars":  
        from models.qwen_uitars_uitars import QwenUITarsTripleComboMethod
        model = QwenUITarsTripleComboMethod ()
    
    elif model_type == "gemini_gemini":  
        from models.gemini_gemini import GeminiGeminiComboMethod
        model = GeminiGeminiComboMethod ()

    elif model_type == "qwen_only":  
        from models.qwen_only import QwenSingleMethod  
        model = QwenSingleMethod ()
    
    elif model_type == "qwen_qwen":  
        from models.qwen_qwen import QwenQwenComboMethod 
        model = QwenQwenComboMethod ()

    elif model_type == "uitars_only":  
        from models.uitars_only import UITarsSingleMethod  
        model = UITarsSingleMethod()

    elif model_type == "internvl3_only":  
        from models.internvl3_only import InternVL3OnlyMethod  
        model = InternVL3OnlyMethod()
        
    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model
