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
  
  
class QwenUITarsTripleComboMethod:  
    def __init__(self, uitars_model="bytedance/ui-tars-1.5-7b",    
                 qwen_model="qwen/qwen2.5-vl-72b-instruct",    
                 api_base="https://openrouter.ai/api/v1/chat/completions"):  
        """    
        三层组合模型: Qwen初检 + UI-TARS修正 + UI-TARS验证修正  
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
      
    def ground_with_qwen_initial(self, instruction, image):  
        """第一层: Qwen初检 - 输出像素坐标"""  
        self.debug_print("=== First layer: Qwen initial detection ===")  
        
        width, height = image.size  
        
        # Qwen 初检提示词
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
        
        response = self.call_openrouter_api(self.qwen_model, prompt, image)  
        self.debug_print(f"Qwen响应: {response}")  
        
        pixel_point = self.parse_pixel_coordinates_raw(response)  
        return pixel_point, response  
      
    def refine_with_uitars(self, instruction, image, qwen_pixel_point, qwen_response):  
        """第二层: UI-TARS修正 - 基于Qwen初检结果进行修正"""  
        self.debug_print("=== Second layer: UI-TARS refinement ===")  
        
        width, height = image.size  
        
        # 将Qwen的像素坐标转换为提示词中的描述  
        x_pixel, y_pixel = qwen_pixel_point if qwen_pixel_point else (width//2, height//2)  
        
        # UI-TARS 修正提示词 - 使用原来的初检提示词格式  
        prompt = f"""You are an expert GUI grounding agent specialized in precisely locating UI elements in screenshots.    
    
    **Task**: Identify and locate the exact center point of the UI element described below.    
    
    **Target Element**: {instruction}    
    
    **Previous detection found coordinate**: [{x_pixel}, {y_pixel}]  
    
    **Analysis Guidelines**:    
    1. **Visual Identification**: Examine text labels, icons, shapes, colors, and visual styling    
    2. **Spatial Context**: Consider the element's position relative to other UI components, containers, and screen regions    
    3. **Disambiguation**: If multiple similar elements exist, select the one that best matches ALL aspects of the instruction (text content, position, context)    
    4. **Precision**: Return the CENTER POINT of the target element's bounding box  
    5. **Refinement**: ONLY keep the previous point if it is unquestionably correct; otherwise, correct it.  
    
    Output the tool call in JSON:    
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}    
    
    Return ONLY the JSON, no explanation."""  
        
        response = self.call_openrouter_api(self.uitars_model, prompt, image)  
        self.debug_print(f"UI-TARS修正响应: {response}")  
        
        refined_point = self.parse_pixel_coordinates_raw(response)  
        return refined_point, response  
    
    def verify_with_uitars(self, instruction, image, refined_pixel_point, refined_response):  
        """第三层: UI-TARS验证修正 - 对比第二层输出与实际视觉进行最终修正"""  
        self.debug_print("=== Third layer: UI-TARS verification and correction ===")  
          
        width, height = image.size  
          
        # 使用第二层的像素坐标  
        x_pixel, y_pixel = refined_pixel_point if refined_pixel_point else (width//2, height//2)  
          
        # 第三层验证提示词 - 强调视觉验证  
        prompt = f"""You are an expert GUI verification agent. Your task is to verify and correct the detected coordinate.  
  
**Target Element**: {instruction}  
  
**Second-layer detection coordinate**: [{x_pixel}, {y_pixel}]  
  
**Verification Protocol**:  
1. **Visual Inspection**: Look at the coordinate [{x_pixel}, {y_pixel}] on the screenshot  
2. **Accuracy Check**:   
   - Is this coordinate EXACTLY at the CENTER of the target element?  
   - If YES: Return the same coordinate  
   - If NO: Identify the correct center and return the corrected coordinate  
  
3. **Common Errors to Check**:  
   - Offset from center (coordinate is on edge/corner instead of center)  
   - Wrong element (coordinate points to a similar but incorrect element)  
   - Boundary issues (coordinate is outside the element bounds)  
  
4. **Correction Guidelines**:  
   - Measure the visual distance between current point and true center  
   - Calculate the corrected pixel position  
   - Ensure the corrected point is within screen bounds: x∈[0,{width}], y∈[0,{height}]  
  
Output the tool call in JSON:  
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
  
Return ONLY the JSON, no explanation."""  
          
        response = self.call_openrouter_api(self.uitars_model, prompt, image)  
        self.debug_print(f"UI-TARS验证修正响应: {response}")  
          
        verified_point = self.parse_pixel_coordinates_raw(response)  
        return verified_point, response  
      
    def ground_only_positive(self, instruction, image):  
        """主入口:执行三层grounding - Qwen初检 + UI-TARS修正 + UI-TARS验证"""  
        self.logs = []  
          
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
          
        width, height = image.size  
          
        # 第一层: Qwen初检  
        qwen_pixel_point, qwen_response = self.ground_with_qwen_initial(instruction, image)  
          
        time.sleep(1)  # 避免速率限制  
          
        # 第二层: UI-TARS修正  
        refined_pixel_point, refined_response = self.refine_with_uitars(  
            instruction, image, qwen_pixel_point, qwen_response  
        )  
          
        time.sleep(1)  # 避免速率限制  
          
        # 第三层: UI-TARS验证修正  
        final_pixel_point, verified_response = self.verify_with_uitars(  
            instruction, image, refined_pixel_point, refined_response  
        )  
          
        # 归一化最终坐标  
        final_normalized_point = self.normalize_pixel_coordinates(final_pixel_point, width, height)  
          
        result_dict = {  
            "result": "positive" if final_normalized_point else "negative",  
            "point": final_normalized_point,  
            "bbox": None,  
            "raw_response": {  
                "qwen_initial": qwen_response,  
                "uitars_refined": refined_response,  
                "uitars_verified": verified_response,  
                "logs": self.logs  
            }  
        }  
          
        return result_dict  
      
    def ground_allow_negative(self, instruction, image):  
        """支持负样本的grounding"""  
        return self.ground_only_positive(instruction, image)