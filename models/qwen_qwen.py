import os  
import json  
import base64  
from io import BytesIO  
from PIL import Image  
import requests  
import time  
import re 
from PIL import ImageDraw  
  
  
def convert_pil_image_to_base64(image):  
    """Convert PIL Image to base64 string."""  
    buffered = BytesIO()  
    image.save(buffered, format="PNG")  
    img_str = base64.b64encode(buffered.getvalue()).decode()  
    return img_str  
  
  
class QwenQwenComboMethod:  
    def __init__(self, qwen_model="qwen/qwen2.5-vl-72b-instruct",  
                 api_base="https://openrouter.ai/api/v1/chat/completions"): 
          
        self.qwen_model = qwen_model  
        self.api_base = api_base  
          
        self.api_key = os.environ.get("OPENROUTER_API_KEY")  
        if not self.api_key:  
            raise ValueError("请设置OPENROUTER_API_KEY环境变量")  
          
        self.override_generation_config = {  
            "temperature": 0.0,  
            "max_tokens": 2048,  
        }  
        self.logs = []  
        self.debug_flag = True  
      
    def set_generation_config(self, **kwargs):  
        self.override_generation_config.update(kwargs)  
        if "max_new_tokens" in self.override_generation_config:  
            self.override_generation_config["max_tokens"] = self.override_generation_config["max_new_tokens"]  
            del self.override_generation_config["max_new_tokens"]  
      
    def debug_print(self, string):  
        self.logs.append(string)  
        if self.debug_flag:  
            print(string)  
      
    def load_model(self, model_name_or_path=None):   
        self.debug_print(f"使用OpenRouter API调用模型: {self.qwen_model} (双层检测)")  
      
    def call_openrouter_api(self, model_name, prompt, image, max_retries=3):  
        headers = {  
            "Authorization": f"Bearer {self.api_key}",  
            "Content-Type": "application/json",  
        }  
          
        image_base64 = convert_pil_image_to_base64(image)  
        messages = [  
            {  
                "role": "user",  
                "content": [  
                    {  
                        "type": "image_url",  
                        "image_url": {  
                            "url": f"data:image/png;base64,{image_base64}"  
                        }  
                    },  
                    {  
                        "type": "text",  
                        "text": prompt  
                    }  
                ]  
            }  
        ]  
          
        payload = {  
            "model": model_name,  
            "messages": messages,  
            "temperature": self.override_generation_config.get("temperature", 0.0),  
            "max_tokens": self.override_generation_config.get("max_tokens", 2048),  
        }  
          
        for attempt in range(max_retries):  
            try:  
                response = requests.post(  
                    self.api_base,  
                    headers=headers,  
                    json=payload,  
                    timeout=60  
                )  
                response.raise_for_status()  
                result = response.json()  
                return result["choices"][0]["message"]["content"]  
            except requests.exceptions.HTTPError as e:  
                status_code = e.response.status_code  
                if status_code == 429:  
                    wait_time = (2 ** attempt) * 2  
                    self.debug_print(f"速率限制,等待{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")  
                    if attempt < max_retries - 1:  
                        time.sleep(wait_time)  
                        continue  
                elif status_code >= 500:  
                    wait_time = (2 ** attempt) * 1  
                    self.debug_print(f"服务器错误({status_code}),等待{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")  
                    if attempt < max_retries - 1:  
                        time.sleep(wait_time)  
                        continue  
                self.debug_print(f"API调用失败: {str(e)}")  
                return None  
            except Exception as e:  
                self.debug_print(f"API调用失败: {str(e)}")  
                if attempt < max_retries - 1:  
                    time.sleep(1)  
                    continue  
                return None  
          
        return None 

    def plot_annotated_circle(self, image, point, radius=15, is_correct=None, alpha=100): 
         
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))  
        draw = ImageDraw.Draw(overlay)  
        
        if point is not None:  
            x, y = point  
            
            
            if is_correct:  
                color = (0, 0, 255, alpha)  
            else:  
                color = (255, 0, 0, alpha)   
            
             
            draw.ellipse(  
                (x - radius, y - radius, x + radius, y + radius),  
                fill=color    
            )  
        
        annotated_image = image.convert('RGBA')  
        annotated_image = Image.alpha_composite(annotated_image, overlay)  
        return annotated_image.convert('RGB')

    def parse_pixel_coordinates_raw(self, response):  
   
        if not response:  
            return None  
        
        try:  
            response = re.sub(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\)', r'[\1, \2]', response)  
            
            if '"coordinate"' in response:  
                open_brackets = response.count('[')  
                close_brackets = response.count(']')  
                open_braces = response.count('{')  
                close_braces = response.count('}')  
                
                if open_brackets > close_brackets:  
                    response += ']' * (open_brackets - close_brackets)  
                if open_braces > close_braces:  
                    response += '}' * (open_braces - close_braces)  
              
            if "coordinate" in response:  
                try:  
                    data = json.loads(response)  
                    coords = data.get("arguments", {}).get("coordinate", [])  
                    if len(coords) == 2:  
                        x_pixel = float(coords[0])  
                        y_pixel = float(coords[1])  
                        self.debug_print(f"提取像素坐标: [{x_pixel}, {y_pixel}]")  
                        return [x_pixel, y_pixel]  
                except json.JSONDecodeError:  
                    pass  
 
            match = re.search(r'"coordinate"\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)', response)  
            if match:  
                x_pixel = float(match.group(1))  
                y_pixel = float(match.group(2))  
                self.debug_print(f"提取像素坐标: [{x_pixel}, {y_pixel}]")  
                return [x_pixel, y_pixel]  
            
            match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', response)  
            if match:  
                x_pixel = float(match.group(1))  
                y_pixel = float(match.group(2))  
                self.debug_print(f"提取像素坐标: [{x_pixel}, {y_pixel}]")  
                return [x_pixel, y_pixel]  
                
        except Exception as e:  
            self.debug_print(f"坐标解析失败: {e}")  
        
        return None 
      
    def normalize_pixel_coordinates(self, pixel_point, width, height):  
 
        if pixel_point is None:  
            return None  
        
        try:  
            x_pixel, y_pixel = pixel_point  
            x_norm = max(0.0, min(1.0, x_pixel / width))  
            y_norm = max(0.0, min(1.0, y_pixel / height))  
            self.debug_print(f"归一化: [{x_pixel}, {y_pixel}] -> [{x_norm:.4f}, {y_norm:.4f}]")  
            return [x_norm, y_norm]  
        except Exception as e:  
            self.debug_print(f"归一化失败: {e}")  
            return None  
    
    def initial_detection(self, instruction, image):  
 
        self.debug_print("=== First layer: Initial detection ===")  
        
        width, height = image.size  
        
        prompt = f"""You are a GUI automation assistant. The screen resolution is {width}x{height}.  
    
    Task: {instruction}  
    
    Use the computer_use tool to click on the target element.  
    
    Available tool:  
    - computer_use(action: str, coordinate: list[int, int])  
    - action: "left_click"  
    - coordinate: [x, y] pixel position where x∈[0,{width}], y∈[0,{height}]  
    
    Output the tool call in JSON:  
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
    
    Return ONLY the JSON, no explanation."""  
        
        response = self.call_openrouter_api(self.qwen_model, prompt, image)  
        self.debug_print(f"Initial response: {response}")  
        
        pixel_point = self.parse_pixel_coordinates_raw(response)  
        return pixel_point, response


    def refinement_detection(self, instruction, image, initial_pixel_point, initial_response, is_correct=False):  

        self.debug_print("=== Second layer: Refinement ===")  
        
        width, height = image.size  
        
        if initial_pixel_point is None:  
            self.debug_print("No initial point to refine")  
            return None, "No initial detection result to refine"  
         
        annotated_image = self.plot_annotated_circle(  
            image,  
            initial_pixel_point,  
            radius=30,  
            is_correct=is_correct,  
            alpha=100  
        )  
        
        x_pixel, y_pixel = initial_pixel_point  
        
        prompt = f"""You are verifying a GUI element detection result.  
    
    **Original Task**: {instruction}  
    
    **Current Detection**: A semi-transparent red circle marks the point [{x_pixel}, {y_pixel}] as the detected target.  
    
    **Your Task**:  
    1. Examine what UI element is at/near the red circle  
    2. Compare it with the task requirement: "{instruction}"  
    3. Decide:  
    - If the marked point IS the correct target → output the SAME coordinate [{x_pixel}, {y_pixel}]  
    - If the marked point is NOT correct → find and output the correct element's center coordinate  
    
    **Critical**:  
    - The red circle shows the PREVIOUS detection point, not necessarily the correct answer  
    - You must verify if the element at this point matches "{instruction}" before deciding  
    - Only change the coordinate if you find a better match  
    
    Output JSON: {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}"""   
        
        response = self.call_openrouter_api(self.qwen_model, prompt, annotated_image)  
        self.debug_print(f"Refinement response: {response}")  
        
        refined_pixel_point = self.parse_pixel_coordinates_raw(response)  
        return refined_pixel_point, response  
      
    def ground_only_positive(self, instruction, image):  
 
        self.logs = []  
        
        if isinstance(image, str):  
            image_path = image  
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."  
            image = Image.open(image_path).convert('RGB')  
        assert isinstance(image, Image.Image), "Invalid input image."  
        
        width, height = image.size  
        
        initial_pixel_point, initial_response = self.initial_detection(instruction, image)  
        
        if initial_pixel_point is None:  
            return {  
                "result": "negative",  
                "point": None,  
                "bbox": None,  
                "raw_response": {"qwen_initial": initial_response, "logs": self.logs}  
            }  
        
        time.sleep(1)  
        
        final_pixel_point, refined_response = self.refinement_detection(  
            instruction, image, initial_pixel_point, initial_response, is_correct=False  
        )  
        
        if final_pixel_point is None:  
            self.debug_print("修正失败,回退到初检结果")  
            final_pixel_point = initial_pixel_point  
         
        final_normalized_point = self.normalize_pixel_coordinates(final_pixel_point, width, height)  
        
        return {  
            "result": "positive" if final_normalized_point else "negative",  
            "point": final_normalized_point,  
            "bbox": None,  
            "raw_response": {  
                "qwen_initial": initial_response,  
                "qwen_refined": refined_response,  
                "logs": self.logs  
            }  
        }  
      
    def ground_allow_negative(self, instruction, image):  
        """支持负样本的grounding"""  
        return self.ground_only_positive(instruction, image)