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
  
  
class UITarsQwenComboMethod:    
    def __init__(self, uitars_model="bytedance/ui-tars-1.5-7b",    
                 qwen_model="qwen/qwen2.5-vl-72b-instruct",    
                 api_base="https://openrouter.ai/api/v1/chat/completions"):    
        """    
        双层组合模型: UI-TARS初检 + Qwen修正    
            
        Args:    
            uitars_model: UI-TARS模型名称    
            qwen_model: Qwen模型名称    
            api_base: OpenRouter API端点    
        """    
        self.uitars_model = uitars_model    
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
        
    def load_model(self):    
        """加载模型(API调用无需实际加载)"""    
        pass    
        
    def set_generation_config(self, **kwargs):    
        self.override_generation_config.update(kwargs)    
        
    def debug_print(self, string):    
        self.logs.append(string)    
        if self.debug_flag:    
            print(string)    
        
    def call_openrouter_api(self, model, prompt, image, max_retries=3):    
        """调用OpenRouter API"""    
        base64_image = convert_pil_image_to_base64(image)    
            
        headers = {    
            "Authorization": f"Bearer {self.api_key}",    
            "Content-Type": "application/json"    
        }    
            
        payload = {    
            "model": model,    
            "messages": [    
                {    
                    "role": "user",    
                    "content": [    
                        {    
                            "type": "text",    
                            "text": prompt    
                        },    
                        {    
                            "type": "image_url",    
                            "image_url": {    
                                "url": f"data:image/png;base64,{base64_image}"    
                            }    
                        }    
                    ]    
                }    
            ],    
            **self.override_generation_config    
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
                return result['choices'][0]['message']['content']    
            except Exception as e:    
                self.debug_print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")    
                if attempt < max_retries - 1:    
                    time.sleep(2 ** attempt)    
                else:    
                    return None    
        
    def parse_pixel_coordinates_raw(self, response):    
        """解析像素坐标但不归一化 - 增强版"""    
        if not response:    
            return None    
          
        try:    
            # 修复常见的 JSON 格式错误    
            # 1. 修复 [x, y) 格式 (括号不匹配)    
            response = re.sub(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\)', r'[\1, \2]', response)    
              
            # 2. 修复不完整的 JSON (添加缺失的闭合括号)    
            if '"coordinate"' in response:    
                open_brackets = response.count('[')    
                close_brackets = response.count(']')    
                open_braces = response.count('{')    
                close_braces = response.count('}')    
                  
                if open_brackets > close_brackets:    
                    response += ']' * (open_brackets - close_brackets)    
                if open_braces > close_braces:    
                    response += '}' * (open_braces - close_braces)    
              
            # 3. 尝试标准 JSON 解析    
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
              
            # 4. 正则表达式提取 (最后的备选方案)    
            match = re.search(r'"coordinate"\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)', response)    
            if match:    
                x_pixel = float(match.group(1))    
                y_pixel = float(match.group(2))    
                self.debug_print(f"提取像素坐标: [{x_pixel}, {y_pixel}]")    
                return [x_pixel, y_pixel]    
              
            # 5. 尝试直接提取 [x, y] 格式    
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
        
    def ground_with_uitars_initial(self, instruction, image):    
        """第一层: UI-TARS初检 - 输出像素坐标"""    
        self.debug_print("=== First layer: UI-TARS initial detection ===")    
          
        width, height = image.size    
          
        # UI-TARS 初检提示词  
        prompt = f"""Task: {instruction}      
Screen: {width}x{height}      
  
Use the computer_use tool to click on the target element.      
  
Available tool:      
- computer_use(action: str, coordinate: list[int, int])      
- action: "left_click"      
- coordinate: [x, y] pixel position where x∈[0,{width}], y∈[0,{height}]    
  
Task:    
- Find the element that best satisfies the instruction.    
- Return the CENTER of that element.    
  
Rules:    
- Prioritize function over appearance.    
- Use the element's clickable area/visual bounds.    
- If multiple candidates exist, choose the most appropriate.      
  
Output JSON: {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}      
x,y in pixels [0-{width}, 0-{height}]"""    
          
        response = self.call_openrouter_api(self.uitars_model, prompt, image)    
        self.debug_print(f"UI-TARS响应: {response}")    
          
        pixel_point = self.parse_pixel_coordinates_raw(response)    
        return pixel_point, response    
        
    def refine_with_qwen(self, instruction, image, uitars_pixel_point, uitars_response):  
        """第二层: Qwen修正 - 基于UI-TARS初检结果进行修正"""  
        self.debug_print("=== Second layer: Qwen refinement ===")  
          
        width, height = image.size  
          
        # 将UI-TARS的像素坐标转换为提示词中的描述  
        x_pixel, y_pixel = uitars_pixel_point if uitars_pixel_point else (width//2, height//2)  
          
        # Qwen修正提示词  
        prompt = f"""Task: {instruction}  
Screen size: {width}x{height}  
Initial detection point: [{x_pixel}, {y_pixel}]  
  
**Correction Guidelines:**  
1. **Only correct if there is a REAL positioning deviation** - do not make arbitrary changes  
2. **Verify accuracy**: Check if the initial point is on the correct UI element  
3. **Precision correction**: If correction is needed, move to the exact center of the target element  
4. **No change if accurate**: Keep the initial point if it's correctly positioned  
  
**Correction Criteria (ONLY correct when):**  
- Point is on wrong UI element  
- Point misses the target element entirely    
- Point is on edge/corner instead of center  
- Point is on disabled/inactive element  
  
Output JSON format:  
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
  
Return ONLY the JSON."""  
          
        response = self.call_openrouter_api(self.qwen_model, prompt, image)  
        self.debug_print(f"Qwen修正响应: {response}")  
          
        refined_point = self.parse_pixel_coordinates_raw(response)  
        return refined_point, response   
      
    def ground_only_positive(self, instruction, image):    
        """主入口:执行双层grounding - UI-TARS初检 + Qwen修正"""    
        self.logs = []    
          
        if isinstance(image, str):    
            image = Image.open(image).convert('RGB')    
          
        width, height = image.size    
          
        # 第一层: UI-TARS初检    
        uitars_pixel_point, uitars_response = self.ground_with_uitars_initial(instruction, image)    
          
        time.sleep(1)  # 避免速率限制    
          
        # 第二层: Qwen修正 (必须基于初检结果)    
        final_pixel_point, qwen_response = self.refine_with_qwen(    
            instruction, image, uitars_pixel_point, uitars_response    
        )    
          
        # 统一归一化 - 无论修正是否成功都使用修正层的输出    
        final_normalized_point = self.normalize_pixel_coordinates(final_pixel_point, width, height)    
          
        result_dict = {    
            "result": "positive" if final_normalized_point else "negative",    
            "point": final_normalized_point,    
            "bbox": None,    
            "raw_response": {    
                "uitars_initial": uitars_response,    
                "qwen_refined": qwen_response,    
                "logs": self.logs    
            }    
        }    
          
        return result_dict    
        
    def ground_allow_negative(self, instruction, image):    
        """支持负样本的grounding"""    
        return self.ground_only_positive(instruction, image)