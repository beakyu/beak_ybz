import os      
import re      
import json      
import base64      
import requests      
from PIL import Image      
import io      
import time    
    
    
def convert_pil_image_to_base64(image):      
    """将PIL图像编码为base64 - 使用JPEG格式"""      
    if isinstance(image, str):      
        image = Image.open(image).convert('RGB')      
    buffer = io.BytesIO()      
    image.save(buffer, format='JPEG', quality=85)      
    return base64.b64encode(buffer.getvalue()).decode('utf-8')      
    
    
class UITarsGeminiComboMethod:      
    """双层组合模型: UI-TARS初检 + Gemini修正"""      
      
    def __init__(self, uitars_model="bytedance/ui-tars-1.5-7b",       
                 gemini_model="google/gemini-2.5-pro",      
                 api_base="https://openrouter.ai/api/v1/chat/completions"):      
        """      
        双层组合模型: UI-TARS初检 + Gemini修正      
              
        Args:      
            uitars_model: UI-TARS模型名称      
            gemini_model: Gemini模型名称        
            api_base: OpenRouter API端点      
        """      
        self.uitars_model = uitars_model      
        self.gemini_model = gemini_model      
        self.api_base = api_base      
              
        self.api_key = os.environ.get("OPENROUTER_API_KEY")      
        if not self.api_key:      
            raise ValueError("请设置OPENROUTER_API_KEY环境变量")      
              
        self.configs = {"max_size": 2048}      
        self.logs = []      
        self.override_generation_config = {      
            "temperature": 0.0,      
            "max_tokens": 4000      
        }      
      
    def load_model(self):      
        """加载模型(API调用无需实际加载)"""      
        pass      
      
    def set_generation_config(self, **kwargs):      
        self.override_generation_config.update(kwargs)      
      
    def debug_print(self, string):      
        self.logs.append(string)      
        print(string)      
      
    def encode_image_to_base64(self, image):      
        """将PIL图像编码为base64"""      
        if isinstance(image, str):      
            image = Image.open(image).convert('RGB')      
        buffer = io.BytesIO()      
        image.save(buffer, format='JPEG', quality=85)      
        return base64.b64encode(buffer.getvalue()).decode('utf-8')      
      
    def call_gemini(self, prompt, image):      
        """调用Gemini API - 使用GeminiOnlyMethod的稳定实现"""      
        base64_image = self.encode_image_to_base64(image)      
      
        headers = {      
            "Authorization": f"Bearer {self.api_key}",      
            "Content-Type": "application/json",      
            "HTTP-Referer": "https://localhost:3000",      
            "X-Title": "ScreenSpot-Pro-Evaluation"      
        }      
      
        data = {      
            "model": self.gemini_model,      
            "messages": [{      
                "role": "user",      
                "content": [      
                    {"type": "text", "text": prompt},      
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}      
                ]      
            }],      
            "max_tokens": self.override_generation_config["max_tokens"],      
            "temperature": 0      
        }      
      
        try:      
            response = requests.post(self.api_base, headers=headers, json=data, timeout=60)      
            if response.status_code == 200:      
                return response.json()['choices'][0]['message']['content']      
            else:      
                raise Exception(f"Gemini API错误: {response.status_code}")      
        except Exception as e:      
            self.debug_print(f"Gemini调用失败: {e}")      
            return None      
      
    def parse_coordinates(self, response_text):      
        """解析JSON坐标结果 - 使用GeminiOnlyMethod的稳定实现"""      
        if not response_text:      
            return []      
      
        coordinates = []      
        patterns = [      
            r'```(?:json)?\s*(\[.*?\])\s*```',      
            r'\[(?:[^[\]]*(?:\{[^}]*\})[^[\]]*)*\]'      
        ]      
      
        for pattern in patterns:      
            matches = re.findall(pattern, response_text, re.DOTALL)      
            for match in matches:      
                try:      
                    parsed = json.loads(match.strip())      
                    if isinstance(parsed, list):      
                        coordinates = parsed      
                        break      
                except json.JSONDecodeError:      
                    continue      
            if coordinates:      
                break      
      
        return coordinates      
      
    def call_uitars(self, prompt, image):      
        """调用UI-TARS API"""      
        base64_image = convert_pil_image_to_base64(image)      
      
        headers = {      
            "Authorization": f"Bearer {self.api_key}",      
            "Content-Type": "application/json",      
            "HTTP-Referer": "https://localhost:3000",      
            "X-Title": "ScreenSpot-Pro-Evaluation"      
        }      
      
        data = {      
            "model": self.uitars_model,      
            "messages": [{      
                "role": "user",      
                "content": [      
                    {"type": "text", "text": prompt},      
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}      
                ]      
            }],      
            "max_tokens": 2048,      
            "temperature": 0      
        }      
      
        try:      
            response = requests.post(self.api_base, headers=headers, json=data, timeout=60)      
            if response.status_code == 200:      
                return response.json()['choices'][0]['message']['content']      
            else:      
                raise Exception(f"UI-TARS API错误: {response.status_code}")      
        except Exception as e:      
            self.debug_print(f"UI-TARS调用失败: {e}")      
            return None      
      
    def parse_uitars_response(self, response, image_size=None):      
        """解析UI-TARS输出格式并归一化坐标"""      
        if not response:      
            return None      
      
        try:      
            # 尝试提取JSON块      
            if "```json" in response:      
                start_idx = response.find("```json") + len("```json")      
                end_idx = response.find("```", start_idx)      
                if end_idx == -1:      
                    json_str = response[start_idx:].strip()      
                else:      
                    json_str = response[start_idx:end_idx].strip()      
            elif "{" in response and "}" in response:      
                start_idx = response.find("{")      
                end_idx = response.rfind("}") + 1      
                json_str = response[start_idx:end_idx]      
            else:      
                json_str = response      
      
            # 尝试修复不完整的JSON      
            if json_str.count('[') > json_str.count(']'):      
                json_str += ']' * (json_str.count('[') - json_str.count(']'))      
            if json_str.count('{') > json_str.count('}'):      
                json_str += '}' * (json_str.count('{') - json_str.count('}'))      
      
            data = json.loads(json_str)      
            coords = data.get("coordinate", None)      
      
            if coords and len(coords) >= 2:      
                x, y = float(coords[0]), float(coords[1])      
      
                # 检测是否为像素坐标(值>1),如果是则归一化      
                if image_size and (x > 1 or y > 1):      
                    width, height = image_size      
                    x_norm = x / width      
                    y_norm = y / height      
                    self.debug_print(f"归一化: [{x}, {y}] -> [{x_norm:.4f}, {y_norm:.4f}] (图像尺寸: {width}x{height})")      
                    x, y = x_norm, y_norm      
      
                # 确保坐标在0-1范围内      
                x = max(0.0, min(1.0, x))      
                y = max(0.0, min(1.0, y))      
      
                return [x, y]      
      
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:      
            self.debug_print(f"坐标解析失败: {e}")      
      
        return None      
      
    def ground_with_uitars(self, instruction, image):      
        """使用UI-TARS进行初检"""      
        self.debug_print("=== UI-TARS初检 ===")      
      
        prompt = f"""Task: {instruction}    
    
Output format: {{"coordinate": [x, y]}}    
    
x, y must be normalized (0-1 range)."""      
      
        response = self.call_uitars(prompt, image)      
        self.debug_print(f"UI-TARS响应: {response}")      
      
        image_size = (image.width, image.height)      
        point = self.parse_uitars_response(response, image_size)      
        return point, response  
      
    def refine_with_gemini(self, instruction, image, uitars_point, uitars_response):      
        """使用Gemini进行修正"""      
        if uitars_point is None:      
            # 如果UI-TARS未找到目标,让Gemini独立检测      
            self.debug_print("=== Gemini独立检测 ===")      
            prompt = f"""You are a GUI element localization expert. Your task is to identify the target UI element described in the instruction and return its center coordinates.      
      
Instruction: {instruction}      
      
**Analysis Steps:**      
1. First, identify the general area where the target element is located (e.g., toolbar, sidebar, main content area)      
2. Then, locate the specific element within that area      
3. Consider neighboring elements to help pinpoint the exact location      
4. Determine the precise center point of the target element      
      
**Output Format:**      
Return a JSON array containing all potentially matching UI elements, ranked by confidence:      
[{{"label": "detailed_element_description", "x": number, "y": number, "confidence": number, "reasoning": "why this matches"}}]      
      
**Requirements:**      
- x, y are the CENTER coordinates of the element, normalized to 0-1 scale      
- x=0 is left edge, x=1.0 is right edge      
- y=0 is top edge, y=1.0 is bottom edge      
- For example, the center of the image would be x=0.5, y=0.5      
- Be VERY precise with coordinates - aim for the exact center of the target element      
- Include detailed labels that describe the element's appearance and context      
- Provide confidence scores (0-1) based on how well the element matches the instruction      
- Include reasoning for each candidate to explain the match      
- Return ONLY the JSON array, no additional text      
      
**Important:**      
- Focus on the EXACT element mentioned in the instruction, not similar elements      
- If the instruction mentions specific text or labels, prioritize elements with that exact text      
- Consider the element's visual appearance (color, size, shape) if mentioned in the instruction"""      
      
            response = self.call_gemini(prompt, image)      
            self.debug_print(f"Gemini响应: {response}")      
      
            if not response:      
                return None, response      
      
            coords = self.parse_coordinates(response)      
            self.debug_print(f"Gemini检测到 {len(coords)} 个元素")      
      
            if not coords:      
                return None, response      
      
            # 验证坐标并选择最佳匹配      
            validated_coords = []      
            for coord in coords:      
                if not isinstance(coord, dict):      
                    continue      
                x = max(0.0, min(1.0, float(coord.get('x', 0))))      
                y = max(0.0, min(1.0, float(coord.get('y', 0))))      
                validated_coords.append({      
                    "label": coord.get('label', 'element'),      
                    "x": x,      
                    "y": y,      
                    "confidence": float(coord.get('confidence', 0.7))      
                })      
      
            if validated_coords:      
                validated_coords.sort(key=lambda x: x['confidence'], reverse=True)      
                best_match = validated_coords[0]      
                point = [best_match['x'], best_match['y']]      
                self.debug_print(f"Gemini独立检测选择: {best_match['label']} at ({point[0]:.3f}, {point[1]:.3f}), confidence={best_match['confidence']:.3f}")      
                return point, response      
      
            return None, response  
          
        # 使用UI-TARS的结果作为参考,让Gemini验证和修正      
        self.debug_print("=== Gemini修正 ===")      
        prompt = f"""You are a GUI element localization expert. Your task is to verify and potentially correct the coordinate provided by a previous detection.    
    
Instruction: {instruction}    
Previous detection: x={uitars_point[0]:.4f}, y={uitars_point[1]:.4f}    
    
**Analysis Steps:**    
1. Examine the screenshot and locate the target UI element    
2. Verify if the previous detection coordinate is accurate    
3. If accurate, confirm it; if not, provide the corrected coordinate    
    
**Output Format:** 
[{{"x": 0.xxx, "y": 0.yyy, "confidence": 0.x, "label": "element_name"}}]  
  
x, y must be normalized (0-1 range)."""  
          
        self.debug_print("=== Gemini修正 ===")  
        response = self.call_gemini(prompt, image)  
        self.debug_print(f"Gemini响应: {response}")  
          
        # 使用parse_coordinates解析Gemini响应  
        coords_list = self.parse_coordinates(response)  
        if coords_list and len(coords_list) > 0:  
            coord = coords_list[0]  
            if isinstance(coord, dict):  
                x = max(0.0, min(1.0, float(coord.get('x', uitars_point[0]))))  
                y = max(0.0, min(1.0, float(coord.get('y', uitars_point[1]))))  
                return [x, y], response  
          
        self.debug_print("Gemini修正失败: 无法解析坐标")  
        return None, response  
  
    def ground_only_positive(self, instruction, image):  
        """主入口:执行双层grounding"""  
        self.logs = []  
          
        # 处理图像输入  
        if isinstance(image, str):  
            image_path = image  
            try:  
                assert os.path.exists(image_path) and os.path.isfile(image_path)  
                image = Image.open(image_path).convert('RGB')  
            except (AssertionError, Exception) as e:  
                error_msg = f"Invalid input image path: {image_path}"  
                self.debug_print(error_msg)  
                return {  
                    "result": "negative",  
                    "point": None,  
                    "bbox": None,  
                    "raw_response": [error_msg]  
                }  
          
        # 第一层: UI-TARS初检  
        uitars_point, uitars_response = self.ground_with_uitars(instruction, image)  
          
        # 添加延迟避免速率限制  
        if uitars_point is not None:  
            time.sleep(1)  
          
        # 第二层: Gemini修正  
        final_point, gemini_response = self.refine_with_gemini(  
            instruction, image, uitars_point, uitars_response  
        )  
          
        # 如果Gemini失败,回退到UI-TARS结果  
        if final_point is None and uitars_point is not None:  
            self.debug_print("Gemini修正失败,使用UI-TARS原始结果")  
            final_point = uitars_point  
          
        result_dict = {  
            "result": "positive" if final_point else "negative",  
            "point": final_point,  
            "bbox": None,  
            "raw_response": {  
                "uitars": uitars_response,  
                "gemini": gemini_response,  
                "logs": self.logs  
            }  
        }  
          
        return result_dict  
  
    def ground_allow_negative(self, instruction, image):  
        """支持负样本的grounding"""  
        return self.ground_only_positive(instruction, image)