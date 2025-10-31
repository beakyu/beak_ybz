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
  
  
class QwenSingleMethod :  
    def __init__(self, qwen_model="qwen/qwen3-vl-32b-instruct",  
                 api_base="https://openrouter.ai/api/v1/chat/completions"):  
        """  
        单层检测模型: Qwen直接检测  
          
        Args:  
            qwen_model: Qwen模型名称  
            api_base: OpenRouter API端点  
        """  
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
        """设置生成配置"""  
        self.override_generation_config.update(kwargs)  
        if "max_new_tokens" in self.override_generation_config:  
            self.override_generation_config["max_tokens"] = self.override_generation_config["max_new_tokens"]  
            del self.override_generation_config["max_new_tokens"]  
      
    def debug_print(self, string):  
        """调试输出"""  
        self.logs.append(string)  
        if self.debug_flag:  
            print(string)  
      
    def load_model(self, model_name_or_path=None):  
        """加载模型(API模式无需实际加载)"""  
        self.debug_print(f"使用OpenRouter API调用模型: {self.qwen_model} (单层检测)")  
      
    def call_openrouter_api(self, model_name, prompt, image, max_retries=3):  
        """调用OpenRouter API,带重试机制"""  
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
      
    def parse_response(self, response, image_size=None):  
        """解析模型输出格式并归一化坐标"""  
        if not response:  
            return None  
          
        try:  
            # 尝试提取JSON块  
            if "```json" in response:  
                start_idx = response.find("```json") + len("```json")  
                end_idx = response.find("```", start_idx)  
                json_str = response[start_idx:end_idx].strip()  
            elif "{" in response and "}" in response:  
                start_idx = response.find("{")  
                end_idx = response.rfind("}") + 1  
                json_str = response[start_idx:end_idx]  
            else:  
                json_str = response  
              
            data = json.loads(json_str)  
              
            # 尝试多种可能的键名  
            coords = None  
            for key in ["coordinate", "point", "click_point", "position"]:  
                if key in data:  
                    coords = data[key]  
                    if isinstance(coords, list) and len(coords) == 2:  
                        break  
              
            # 尝试bbox格式  
            if coords is None and "bbox" in data:  
                bbox = data["bbox"]  
                if isinstance(bbox, list) and len(bbox) == 4:  
                    coords = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]  
              
            # 尝试直接解析为数组  
            if coords is None and isinstance(data, list) and len(data) == 2:  
                coords = data  
              
            # 归一化坐标  
            if coords is not None:  
                x, y = float(coords[0]), float(coords[1])  
                  
                if x > 1 or y > 1:  
                    if image_size is None:  
                        self.debug_print(f"警告: 检测到像素坐标 [{x}, {y}] 但未提供图像尺寸,无法归一化")  
                        return None  
                      
                    width, height = image_size  
                    x_norm = max(0.0, min(1.0, x / width))  
                    y_norm = max(0.0, min(1.0, y / height))  
                      
                    self.debug_print(f"归一化: [{x}, {y}] -> [{x_norm:.4f}, {y_norm:.4f}] (图像尺寸: {width}x{height})")  
                    return [x_norm, y_norm]  
                else:  
                    return [x, y]  
          
        except json.JSONDecodeError:  
            # 尝试正则表达式提取坐标  
            match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', response)  
            if match:  
                x, y = float(match.group(1)), float(match.group(2))  
                if (x > 1 or y > 1) and image_size:  
                    width, height = image_size  
                    return [max(0.0, min(1.0, x/width)), max(0.0, min(1.0, y/height))]  
                return [x, y]  
              
            match = re.search(r'x\s*[=:]\s*(\d+\.?\d*),?\s*y\s*[=:]\s*(\d+\.?\d*)', response, re.IGNORECASE)  
            if match:  
                x, y = float(match.group(1)), float(match.group(2))  
                if (x > 1 or y > 1) and image_size:  
                    width, height = image_size  
                    return [max(0.0, min(1.0, x/width)), max(0.0, min(1.0, y/height))]  
                return [x, y]  
          
        return None
      
    def ground_only_positive(self, instruction, image):  
        self.logs = []  
        
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
        
        # 获取图像尺寸用于提示  
        width, height = image.size  
        
        # 更详细的系统提示  
        prompt = f"""You are a precise visual assistant. You will be given an image and an instruction. Your task is to locate the specific element mentioned in the instruction within the image.

Instruction: {instruction}

Please analyze the image and identify the target element. Once located, output a tool call to click on it using the 'computer_use' function.

The 'computer_use' function has the following specification:
- Name: computer_use
- Arguments:
  - action (string): "left_click"
  - coordinate (array of two integers): [x, y] pixel coordinates where x is the horizontal pixel (0 to {width-1}) and y is the vertical pixel (0 to {height-1}). The top-left corner is (0, 0).

Respond with ONLY the JSON-formatted tool call. Do not include any other text or explanation.
Example response:
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x_pixel, y_pixel]}}}}""" 
        
        response = self.call_openrouter_api(self.qwen_model, prompt, image)  
        self.debug_print(f"Qwen响应: {response}")  
        
        # 解析像素坐标并归一化  
        point = self.parse_pixel_coordinates(response, width, height)  
        
        return {  
            "result": "positive" if point else "negative",  
            "point": point,  
            "bbox": None,  
            "raw_response": {"qwen": response, "logs": self.logs}  
        }  
    
    def parse_pixel_coordinates(self, response, width, height):  
        """解析像素坐标并归一化"""  
        try:  
            # 提取 [x, y] 格式  
            match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', response)  
            if match:  
                x_pixel = float(match.group(1))  
                y_pixel = float(match.group(2))  
                
                # 归一化到 [0,1]  
                x_norm = max(0.0, min(1.0, x_pixel / width))  
                y_norm = max(0.0, min(1.0, y_pixel / height))  
                
                self.debug_print(f"像素坐标: [{x_pixel}, {y_pixel}] -> 归一化: [{x_norm:.4f}, {y_norm:.4f}]")  
                return [x_norm, y_norm]  
        except Exception as e:  
            self.debug_print(f"坐标解析失败: {e}")  
        
        return None 
    
      
    def ground_allow_negative(self, instruction, image):  
        """支持负样本的grounding"""  
        return self.ground_only_positive(instruction, image)