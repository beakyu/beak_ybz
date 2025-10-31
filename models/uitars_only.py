import os  
import json  
import base64  
from io import BytesIO  
from PIL import Image  
import requests  
import time  
import re  
  
  
def convert_pil_image_to_base64(image):  
    """Convert PIL Image to base64 string."""  
    buffered = BytesIO()  
    image.save(buffered, format="PNG")  
    img_str = base64.b64encode(buffered.getvalue()).decode()  
    return img_str  
  
  
class UITarsSingleMethod:  
    def __init__(self, uitars_model="bytedance/ui-tars-1.5-7b",  
                 api_base="https://openrouter.ai/api/v1/chat/completions"):  
        """  
        单层检测模型: UI-TARS直接检测  
          
        Args:  
            uitars_model: UI-TARS模型名称  
            api_base: OpenRouter API端点  
        """  
        self.uitars_model = uitars_model  
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
        self.debug_print(f"使用OpenRouter API调用模型: {self.uitars_model} (单层检测)")  
      
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
                
                # 添加字段检查  
                if "choices" not in result or not result["choices"]:  
                    self.debug_print(f"API返回格式异常: {result}")  
                    return None  
                
                return result["choices"][0]["message"]["content"]  
            except requests.exceptions.HTTPError as e:  
                status_code = e.response.status_code  
                if status_code == 429:  
                    wait_time = (2 ** attempt) * 2  
                    self.debug_print(f"遇到速率限制(429),等待{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")  
                    if attempt < max_retries - 1:  
                        time.sleep(wait_time)  
                        continue  
                elif status_code >= 500:  
                    wait_time = (2 ** attempt) * 1  
                    self.debug_print(f"遇到服务器错误({status_code}),等待{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")  
                    if attempt < max_retries - 1:  
                        time.sleep(wait_time)  
                        continue  
                self.debug_print(f"API调用失败: {str(e)}")  
                return None  
            except KeyError as e:  
                self.debug_print(f"API响应缺少必需字段: {e}")  
                return None  
            except Exception as e:  
                self.debug_print(f"API调用失败: {str(e)}")  
                return None  
        
        return None  
      
    def fix_incomplete_json(self, response):  
        """修复不完整的 JSON 响应""" 
        if response is None:  # 添加 None 检查  
            return None 
        if '"coordinate": [' in response:  
            open_brackets = response.count('[')  
            close_brackets = response.count(']')  
            open_braces = response.count('{')  
            close_braces = response.count('}')  
            
            if open_brackets > close_brackets:  
                response += ']' * (open_brackets - close_brackets)  
            if open_braces > close_braces:  
                response += '}' * (open_braces - close_braces)  
        
        return response  
    
    def parse_pixel_coordinates_raw(self, response):  
        """解析像素坐标 - 增强版"""  
        # 先尝试修复 JSON
        if response is None:  # 添加 None 检查  
            return None  
        response = self.fix_incomplete_json(response)  
        
        try:  
            # 标准 JSON 解析  
            if "coordinate" in response:  
                data = json.loads(response)  
                coords = data.get("arguments", {}).get("coordinate", [])  
                if len(coords) == 2:  
                    return [float(coords[0]), float(coords[1])]  
        except json.JSONDecodeError:  
            pass  
        
        try:  
            # 正则表达式提取  
            match = re.search(r'"coordinate"\s*:\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', response)  
            if match:  
                return [float(match.group(1)), float(match.group(2))]  
        except Exception as e:  
            self.debug_print(f"解析失败: {e}")  
        
        return None

    def normalize_1000_coordinates(self, coords_1000):  
        """将0-1000范围的坐标转换为[0,1]范围"""  
        if coords_1000 is None:  
            return None  
        
        try:  
            x_1000, y_1000 = coords_1000  
            # 转换为[0,1]范围  
            x_norm = max(0.0, min(1.0, x_1000 / 1000.0))  
            y_norm = max(0.0, min(1.0, y_1000 / 1000.0))  
            self.debug_print(f"归一化: [{x_1000}, {y_1000}] -> [{x_norm:.4f}, {y_norm:.4f}]")  
            return [x_norm, y_norm]  
        except Exception as e:  
            self.debug_print(f"归一化失败: {e}")  
            return None 
      
    def normalize_pixel_coordinates(self, pixel_point, width, height):  
        """将像素坐标归一化到 [0,1] 范围"""  
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
      
    def ground_only_positive(self, instruction, image):  
        """单层检测入口 - 输出0-1000范围坐标"""  
        self.logs = []  
        
        if isinstance(image, str):  
            image_path = image  
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."  
            image = Image.open(image_path).convert('RGB')  
        assert isinstance(image, Image.Image), "Invalid input image."  
        
        width, height = image.size  
        
        # 修改提示词,要求输出0-1000范围的坐标  
        prompt = f"""You are a GUI verifier. An image is provided.  
    
    Instruction: {instruction}  
    
    Use the computer_use tool to click on the target element.  
    
    Available tool:  
    - computer_use(action: str, coordinate: list[int, int])  
    - action: "left_click"  
    - coordinate: [x, y] where x∈[0,1000], y∈[0,1000] (normalized coordinates)  
    
    Analysis Guidelines:  
    - Visual Identification: Examine text labels, icons, shapes, colors, and visual styling  
    - Spatial Context: Consider the element's position relative to other UI components  
    - Disambiguation: If multiple similar elements exist, select the one that best matches ALL aspects  
    - Precision: Return the CENTER POINT of the target element's bounding box  
    
    Output the tool call in JSON:  
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
    
    Return ONLY the JSON, no explanation."""  
        
        self.debug_print("=== UI-TARS检测 ===")  
        response = self.call_openrouter_api(self.uitars_model, prompt, image)  
        self.debug_print(f"UI-TARS响应: {response}")  
        
        # 解析0-1000范围的坐标  
        coords_1000 = self.parse_pixel_coordinates_raw(response)  
        
        # 转换为[0,1]范围  
        normalized_point = self.normalize_1000_coordinates(coords_1000)  
        
        result_dict = {  
            "result": "positive" if normalized_point else "negative",  
            "point": normalized_point,  
            "bbox": None,  
            "raw_response": {  
                "uitars": response,  
                "logs": self.logs  
            }  
        }  
        
        return result_dict  
      
    def ground_allow_negative(self, instruction, image):  
        return self.ground_only_positive(instruction, image)