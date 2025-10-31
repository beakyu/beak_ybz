import os  
import re  
import json  
import base64  
import requests  
from PIL import Image  
import io  
  
class InternVL3GeminiMethod:  
    """两层检测方法: InternVL3初始检测 + Gemini修正"""  
  
    def __init__(self, planner=None, grounder=None, configs=None):  
        self.internvl3_api_key = os.environ.get("INTERNVL3_API_KEY")  
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")  
  
        if not self.internvl3_api_key or not self.openrouter_api_key:  
            raise ValueError("请设置 INTERNVL3_API_KEY 和 OPENROUTER_API_KEY 环境变量")  
  
        self.internvl3_base_url = "https://chat.intern-ai.org.cn/api/v1/chat/completions"  
        self.internvl3_model = "internvl3-latest"  
        self.gemini_base_url = "https://openrouter.ai/api/v1/chat/completions"  
        self.gemini_model = "google/gemini-2.5-pro"  
  
        self.configs = configs if configs else {"max_size": 2048}  
        self.logs = []  
        self.override_generation_config = {  
            "temperature": 0.0,  
            "max_tokens": 4000  
        }  
  
    def set_generation_config(self, **kwargs):  
        self.override_generation_config.update(kwargs)  
  
    def debug_print(self, string):  
        self.logs.append(string)  
        print(string)  
  
    def encode_image_to_base64(self, image):  
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
        buffer = io.BytesIO()  
        image.save(buffer, format='JPEG', quality=85)  
        return base64.b64encode(buffer.getvalue()).decode('utf-8')  
  
    def call_internvl3(self, prompt, image):  
        base64_image = self.encode_image_to_base64(image)  
        headers = {  
            "Authorization": f"Bearer {self.internvl3_api_key}",  
            "Content-Type": "application/json"  
        }  
        data = {  
            "model": self.internvl3_model,  
            "messages": [{  
                "role": "user",  
                "content": [  
                    {"type": "text", "text": prompt},  
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}  
                ]  
            }],  
            "max_tokens": self.override_generation_config["max_tokens"],  
            "temperature": self.override_generation_config["temperature"],  
            "stream": False  
        }  
        try:  
            response = requests.post(self.internvl3_base_url, headers=headers, json=data, timeout=60)  
            if response.status_code == 200:  
                return response.json()['choices'][0]['message']['content']  
            else:  
                raise Exception(f"InternVL3 API错误: {response.status_code}")  
        except Exception as e:  
            self.debug_print(f"InternVL3调用失败: {e}")  
            return None  
  
    def call_gemini(self, prompt, image):  
        base64_image = self.encode_image_to_base64(image)  
        headers = {  
            "Authorization": f"Bearer {self.openrouter_api_key}",  
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
            response = requests.post(self.gemini_base_url, headers=headers, json=data, timeout=60)  
            if response.status_code == 200:  
                return response.json()['choices'][0]['message']['content']  
            else:  
                raise Exception(f"Gemini API错误: {response.status_code}")  
        except Exception as e:  
            self.debug_print(f"Gemini调用失败: {e}")  
            return None  
  
    def parse_coordinates(self, response_text):  
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
  
    def ground_only_positive(self, instruction, image):      
        """主入口: InternVL3初检 + Gemini迭代修正"""      
        self.logs = []      
            
        if isinstance(image, str):      
            image_path = image      
            if not os.path.exists(image_path):      
                error_msg = f"图像文件不存在: {image_path}"      
                self.debug_print(error_msg)      
                return {"result": "negative", "bbox": None, "point": None, "raw_response": [error_msg]}      
            image = Image.open(image_path).convert('RGB')      
    
        self.debug_print(f"开始迭代检测(InternVL3初检 + Gemini迭代修正): {instruction}")      
    
        # 第一层: InternVL3初始检测  
        initial_prompt = f"""Locate the EXACT CENTER of this UI element: {instruction}    
    
    CRITICAL: Your coordinates must be INSIDE the element's bounding box.    
    - For small elements (icons, buttons): Aim for pixel-perfect center    
    - For text: Target the text center, not container edges    
    - Verify your coordinates would fall WITHIN the element boundaries       
    
    Coordinates:    
    - x, y: EXACT center point (0-1 normalized)    
    - Use 3+ decimals for precision (e.g., 0.234)    
    - x=0 is left, x=1 is right    
    - y=0 is top, y=1 is bottom    
    
    Return ONLY the JSON array."""     
    
        internvl3_response = self.call_internvl3(initial_prompt, image)      
        if not internvl3_response:      
            return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs}      
    
        initial_coords = self.parse_coordinates(internvl3_response)      
        self.debug_print(f"InternVL3检测到 {len(initial_coords)} 个元素")      
    
        if not initial_coords:      
            return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs}      
    
        # 选择初始最佳候选  
        validated_initial = []  
        for coord in initial_coords:  
            if not isinstance(coord, dict):  
                continue  
            x = max(0.0, min(1.0, float(coord.get('x', 0.5))))  
            y = max(0.0, min(1.0, float(coord.get('y', 0.5))))  
            conf = float(coord.get('confidence', 0.5))  
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:  
                validated_initial.append({  
                    "label": coord.get('label', 'element'),  
                    "x": x,  
                    "y": y,  
                    "confidence": conf  
                })  
        
        if not validated_initial:  
            return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs}  
        
        validated_initial.sort(key=lambda x: x['confidence'], reverse=True)  
        current_coords = [validated_initial[0]]  # 只保留最佳候选  
        
        # 第二层: Gemini迭代修正  
        max_iterations = 3  # 最大迭代次数  
        convergence_threshold = 0.01  # 收敛阈值(坐标变化小于此值则停止)  
        
        for iteration in range(max_iterations):  
            self.debug_print(f"--- 迭代 {iteration + 1}/{max_iterations} ---")  
            
            correction_prompt = f"""Review and REFINE these coordinates: {json.dumps(current_coords)}    
    
    Task: {instruction}    
    
    Current prediction: x={current_coords[0]['x']:.4f}, y={current_coords[0]['y']:.4f}  
    
    Your task:  
    1. Verify if this coordinate is at the EXACT CENTER of the target element  
    2. If NOT exact, provide a MORE PRECISE coordinate adjustment  
    3. If already precise, return the same coordinate with high confidence (>0.9)  
    4. Ensure the refined coordinate falls WITHIN the element boundaries  
    
    Return refined JSON: [{{"x": 0.xxx, "y": 0.yyy, "confidence": 0.x, "adjustment_needed": true/false}}]  
    
    **Critical**:   
    - Use 4+ decimals for maximum precision (e.g., 0.2345)  
    - Set adjustment_needed=false if current coordinate is already precise  
    - Set adjustment_needed=true if refinement is needed  
    
    Return ONLY the JSON array."""  
    
            gemini_response = self.call_gemini(correction_prompt, image)  
            if not gemini_response:  
                self.debug_print(f"Gemini调用失败,停止迭代")  
                break  
            
            refined_coords = self.parse_coordinates(gemini_response)  
            if not refined_coords or not isinstance(refined_coords[0], dict):  
                self.debug_print(f"Gemini解析失败,停止迭代")  
                break  
            
            # 验证修正后的坐标  
            refined = refined_coords[0]  
            new_x = max(0.0, min(1.0, float(refined.get('x', current_coords[0]['x']))))  
            new_y = max(0.0, min(1.0, float(refined.get('y', current_coords[0]['y']))))  
            new_conf = float(refined.get('confidence', 0.5))  
            adjustment_needed = refined.get('adjustment_needed', True)  
            
            # 计算坐标变化量  
            delta_x = abs(new_x - current_coords[0]['x'])  
            delta_y = abs(new_y - current_coords[0]['y'])  
            total_delta = (delta_x**2 + delta_y**2)**0.5  
            
            self.debug_print(f"修正: ({current_coords[0]['x']:.4f}, {current_coords[0]['y']:.4f}) -> ({new_x:.4f}, {new_y:.4f})")  
            self.debug_print(f"变化量: {total_delta:.6f}, 置信度: {new_conf:.3f}, 需要调整: {adjustment_needed}")  
            
            # 更新当前坐标  
            current_coords = [{  
                "label": refined.get('label', current_coords[0]['label']),  
                "x": new_x,  
                "y": new_y,  
                "confidence": new_conf  
            }]  
            
            # 收敛判断  
            if total_delta < convergence_threshold or not adjustment_needed or new_conf > 0.9:  
                self.debug_print(f"收敛条件满足,停止迭代 (delta={total_delta:.6f}, conf={new_conf:.3f})")  
                break  
        
        # 最终结果  
        if current_coords:  
            best_match = current_coords[0]  
            point = [best_match['x'], best_match['y']]  
            self.debug_print(f"最终选择: {best_match['label']} at ({point[0]:.4f}, {point[1]:.4f}), confidence={best_match['confidence']:.3f}")  
            return {"result": "positive", "bbox": None, "point": point, "raw_response": self.logs}      
    
        return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs}  
  
    def ground_allow_negative(self, instruction, image):  
        raise NotImplementedError("InternVL3GeminiMethod暂不支持负样本检测")