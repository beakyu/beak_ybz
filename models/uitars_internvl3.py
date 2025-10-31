import os    
import json    
import base64    
from io import BytesIO    
from PIL import Image    
import requests    
import time   
  
  
def convert_pil_image_to_base64(image):    
    """Convert PIL Image to base64 string."""    
    buffered = BytesIO()    
    image.save(buffered, format="PNG")    
    img_str = base64.b64encode(buffered.getvalue()).decode()    
    return img_str    
  
  
class UITarsInternVL3ComboMethod:    
    def __init__(self, uitars_model="bytedance/ui-tars-1.5-7b",     
                 internvl3_model="opengvlab/internvl3-78b",    
                 api_base="https://openrouter.ai/api/v1/chat/completions"):    
        """    
        双层组合模型: UI-TARS初检 + InternVL3修正    
            
        Args:    
            uitars_model: UI-TARS模型名称    
            internvl3_model: InternVL3模型名称      
            api_base: OpenRouter API端点    
        """    
        self.uitars_model = uitars_model    
        self.internvl3_model = internvl3_model    
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
        self.debug_print(f"使用OpenRouter API调用模型: {self.uitars_model} + {self.internvl3_model}")    
  
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
            except Exception as e:    
                self.debug_print(f"API调用失败: {str(e)}")    
                return None    
          
        return None    
  
    def parse_response(self, response, image_size=None):    
        if not response:    
            return None    
          
        try:    
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
              
            coords = None    
            for key in ["coordinate", "point", "click_point", "position"]:    
                if key in data:    
                    coords = data[key]    
                    if isinstance(coords, list) and len(coords) == 2:    
                        break    
              
            if coords is None and "bbox" in data:    
                bbox = data["bbox"]    
                if isinstance(bbox, list) and len(bbox) == 4:    
                    coords = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]    
              
            if coords is None and isinstance(data, list) and len(data) == 2:    
                coords = data    
              
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
            import re    
            match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', response)    
            if match:    
                x, y = float(match.group(1)), float(match.group(2))    
                if (x > 1 or y > 1) and image_size:    
                    width, height = image_size    
                    return [max(0.0, min(1.0, x/width)), max(0.0, min(1.0, y/height))]    
                return [x, y]    
          
        return None    
  
    def ground_with_uitars(self, instruction, image):    
        prompt = f"""You are an expert GUI grounding agent specialized in identifying UI elements in screenshots.    
  
**Task**: Locate the UI element described in the instruction and return its precise center point coordinates.    
  
**Instruction**: {instruction}    
  
**Requirements**:    
1. Carefully analyze the entire screenshot to identify the target UI element    
2. Consider the element's visual characteristics: text labels, icons, colors, position, and surrounding context    
3. If multiple similar elements exist, choose the one that best matches the instruction    
4. Return the CENTER POINT of the target element (not corners or edges)    
5. Coordinates must be normalized to [0,1] range where (0,0) is top-left and (1,1) is bottom-right    
  
**Output Format** (JSON only, no explanation):    
{{"coordinate": [x, y]}}    
  
Where x and y are floating-point numbers between 0 and 1."""    
      
        self.debug_print("=== UI-TARS初检 ===")    
        response = self.call_openrouter_api(self.uitars_model, prompt, image)    
        self.debug_print(f"UI-TARS响应: {response}")    
          
        image_size = (image.width, image.height)    
        point = self.parse_response(response, image_size)    
        return point, response    
  
    def refine_with_internvl3(self, instruction, image, uitars_point, uitars_response):    
        if uitars_point is None:    
            prompt = f"""You are an expert GUI verification agent. The previous model failed to locate the target UI element.    
  
**Task**: Carefully analyze the screenshot and locate the target UI element.    
  
**Instruction**: {instruction}    
  
**Analysis Steps**:    
1. Systematically scan the entire screenshot    
2. Identify all UI elements that could potentially match the instruction    
3. Evaluate each candidate based on visual appearance, functional context, and spatial location    
4. Select the most appropriate match    
5. Return the precise CENTER POINT coordinates    
  
**Output Format** (JSON only):    
{{"coordinate": [x, y], "confidence": "high/medium/low", "reasoning": "brief explanation"}}    
  
Coordinates must be normalized [0,1] range."""    
        else:    
            prompt = f"""You are an expert GUI verification agent. A previous model identified a UI element, but we need your expert verification.    
  
**Original Instruction**: {instruction}    
  
**Previous Model's Prediction**: [{uitars_point[0]:.4f}, {uitars_point[1]:.4f}]    
  
**Your Task**:    
1. Locate the UI element at the predicted coordinate    
2. Verify if it matches the instruction    
3. If accurate, confirm it; if incorrect or imprecise, provide the corrected coordinate    
  
**Critical Considerations**:    
- The coordinate should point to the CENTER of the target element    
- Consider the element's clickable area and visual boundaries    
- Account for any ambiguity in the instruction    
  
**Output Format** (JSON only):    
{{"coordinate": [x, y], "confidence": "high/medium/low", "reasoning": "detailed explanation"}}    
  
Coordinates must be normalized [0,1] range."""    
      
        self.debug_print("=== InternVL3修正 ===")    
        response = self.call_openrouter_api(self.internvl3_model, prompt, image)    
        self.debug_print(f"InternVL3响应: {response}")    
          
        image_size = (image.width, image.height)    
        refined_point = self.parse_response(response, image_size)    
        return refined_point, response    
  
    def ground_only_positive(self, instruction, image):    
        self.logs = []    
          
        if isinstance(image, str):    
            image_path = image    
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."    
            image = Image.open(image_path).convert('RGB')    
        assert isinstance(image, Image.Image), "Invalid input image."    
          
        uitars_point, uitars_response = self.ground_with_uitars(instruction, image)    
          
        if uitars_point is not None:    
            time.sleep(1)    
          
        final_point, internvl3_response = self.refine_with_internvl3(    
            instruction, image, uitars_point, uitars_response    
        )    
          
        if final_point is None and uitars_point is not None:    
            self.debug_print("InternVL3修正失败,使用UI-TARS原始结果")    
            final_point = uitars_point    
          
        result_dict = {    
            "result": "positive" if final_point else "negative",    
            "point": final_point,    
            "bbox": None,    
            "raw_response": {    
                "uitars": uitars_response,    
                "internvl3": internvl3_response,    
                "logs": self.logs    
            }    
        }    
          
        return result_dict    
  
    def ground_allow_negative(self, instruction, image):    
        return self.ground_only_positive(instruction, image)