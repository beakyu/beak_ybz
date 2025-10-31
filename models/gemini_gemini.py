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
    
    
class GeminiGeminiComboMethod:      
    """双层组合模型: Gemini初检 + Gemini修正(自我验证)"""      
      
    def __init__(self, gemini_model="google/gemini-2.5-pro",      
                 api_base="https://openrouter.ai/api/v1/chat/completions"):      
        """      
        双层组合模型: Gemini初检 + Gemini修正(自我验证)      
              
        Args:      
            gemini_model: Gemini模型名称        
            api_base: OpenRouter API端点      
        """      
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
      
    def ground_with_gemini_initial(self, instruction, image):      
        """使用Gemini进行初检"""      
        self.debug_print("=== Gemini初检 ===")      
      
        # 初检prompt - 广泛搜索  
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
        self.debug_print(f"Gemini初检响应: {response}")      
      
        if not response:      
            return None, response      
      
        coords = self.parse_coordinates(response)      
        self.debug_print(f"Gemini初检检测到 {len(coords)} 个元素")      
      
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
            self.debug_print(f"Gemini初检选择: {best_match['label']} at ({point[0]:.3f}, {point[1]:.3f}), confidence={best_match['confidence']:.3f}")      
            return point, response      
      
        return None, response      
      
    def refine_with_gemini(self, instruction, image, initial_point, initial_response):      
        """使用Gemini进行修正(自我验证)"""      
        if initial_point is None:      
            # 如果初检未找到目标,让Gemini重新独立检测      
            self.debug_print("=== Gemini重新检测 ===")      
            return self.ground_with_gemini_initial(instruction, image)  
          
        # 使用初检结果作为参考,让Gemini验证和修正      
        self.debug_print("=== Gemini修正 ===")      
        prompt = f"""You are a GUI element localization expert performing a verification and refinement task.  
  
**Original Task**: {instruction}  
  
**Previous Detection Result**:   
- Coordinate: x={initial_point[0]:.4f}, y={initial_point[1]:.4f}  
  
**Your Task**:  
Critically evaluate the previous detection result. Verify if the coordinate accurately points to the center of the target UI element.  
  
**Verification Steps**:  
1. Locate the target UI element in the screenshot  
2. Compare the previous coordinate with the actual element location  
3. If the coordinate is accurate (within acceptable tolerance), confirm it  
4. If the coordinate is inaccurate or imprecise, provide a corrected coordinate  
  
**Output Format**:  
Return a JSON array with the refined result:  
[{{"label": "element_description", "x": number, "y": number, "confidence": number, "reasoning": "detailed explanation of verification and any corrections"}}]  
  
**Requirements**:  
- x, y must be normalized (0-1 range)  
- Provide high confidence (>0.8) only if you're certain the coordinate is accurate  
- Include detailed reasoning explaining your verification process  
- If you made corrections, explain what was wrong and how you fixed it  
- Return ONLY the JSON array, no additional text  
  
**Critical Considerations**:  
- The coordinate should point to the CENTER of the target element  
- Consider the element's clickable area and visual boundaries  
- Account for any ambiguity in the instruction  
- Prioritize functional correctness over visual similarity"""  
          
        response = self.call_gemini(prompt, image)  
        self.debug_print(f"Gemini修正响应: {response}")  
          
        if not response:  
            self.debug_print("Gemini修正失败: 无响应")  
            return None, response  
          
        coords = self.parse_coordinates(response)  
        self.debug_print(f"Gemini修正检测到 {len(coords)} 个元素")  
          
        if not coords:  
            self.debug_print("Gemini修正失败: 无法解析坐标")  
            return None, response  
          
        # 验证并选择最佳匹配  
        validated_coords = []  
        for coord in coords:  
            if not isinstance(coord, dict):  
                continue  
            x = max(0.0, min(1.0, float(coord.get('x', initial_point[0]))))  
            y = max(0.0, min(1.0, float(coord.get('y', initial_point[1]))))  
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
            self.debug_print(f"Gemini修正选择: {best_match['label']} at ({point[0]:.3f}, {point[1]:.3f}), confidence={best_match['confidence']:.3f}")  
            return point, response  
          
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
                    "bbox": None,    
                    "point": None,    
                    "raw_response": [error_msg]    
                }    
            
        assert isinstance(image, Image.Image), "Invalid input image."    
            
        # 第一层: Gemini初检    
        initial_point, initial_response = self.ground_with_gemini_initial(instruction, image)    
            
        # 添加延迟避免速率限制    
        if initial_point is not None:    
            time.sleep(1)    
            
        # 第二层: Gemini修正(自我验证)  
        final_point, refined_response = self.refine_with_gemini(    
            instruction, image, initial_point, initial_response    
        )    
            
        # 如果修正也失败,回退到初检结果    
        if final_point is None and initial_point is not None:    
            self.debug_print("Gemini修正失败,使用初检原始结果")    
            final_point = initial_point    
            
        result_dict = {    
            "result": "positive" if final_point else "negative",    
            "point": final_point,    
            "bbox": None,    
            "raw_response": {    
                "gemini_initial": initial_response,    
                "gemini_refined": refined_response,    
                "logs": self.logs    
            }    
        }
        return result_dict    
    
    def ground_allow_negative(self, instruction, image):    
        """支持负样本的grounding"""    
        return self.ground_only_positive(instruction, image)