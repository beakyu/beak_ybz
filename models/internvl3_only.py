import os  
import re  
import json  
import base64  
import requests  
from PIL import Image  
import io  
  
class InternVL3OnlyMethod:  
    """单独使用 InternVL3 进行检测"""  
  
    def __init__(self, planner=None, grounder=None, configs=None):  
        self.internvl3_api_key = os.environ.get("INTERNVL3_API_KEY")  
        if not self.internvl3_api_key:  
            raise ValueError("请设置 INTERNVL3_API_KEY 环境变量")  
  
        self.internvl3_base_url = "https://chat.intern-ai.org.cn/api/v1/chat/completions"  
        self.internvl3_model = "internvl3-latest"  
  
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
        """将PIL图像编码为base64"""  
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
        buffer = io.BytesIO()  
        image.save(buffer, format='JPEG', quality=85)  
        return base64.b64encode(buffer.getvalue()).decode('utf-8')  
  
    def call_internvl3(self, prompt, image):  
        """调用InternVL3 API"""  
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
  
    def parse_coordinates(self, response_text):  
        """解析JSON坐标结果"""  
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
        """主入口: 仅使用 InternVL3 进行检测"""  
        self.logs = []  
        
        if isinstance(image, str):  
            image_path = image  
            if not os.path.exists(image_path):  
                error_msg = f"图像文件不存在: {image_path}"  
                self.debug_print(error_msg)  
                return {  
                    "result": "negative",  
                    "bbox": None,  
                    "point": None,  
                    "raw_response": [error_msg]  
                }  
            image = Image.open(image_path).convert('RGB')  
        
        # 获取图像尺寸用于提示词和归一化  
        width, height = image.size  
        
        self.debug_print(f"InternVL3单独检测: {instruction}")  
        
        # 修改后的 prompt - 要求输出像素坐标  
        prompt = f"""You are an expert GUI grounding agent specialized in precisely locating UI elements in screenshots.  
  
**Task**: Identify and locate the exact center point of the UI element described below.  
  
**Target Element**: {instruction} 

Available tool:  
- computer_use(action: str, coordinate: list[int, int])  
  - action: "left_click"  
  - coordinate: [x, y] pixel position where x∈[0,{width}], y∈[0,{height}]  
   
    
    **Output Format:**    
    Return a JSON array containing all potentially matching UI elements, ranked by confidence:    
    [{{"label": "detailed_element_description", "x": number, "y": number, "confidence": number, "reasoning": "why this matches"}}]    
        """  
        
        response = self.call_internvl3(prompt, image)  
        if not response:  
            return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs}  
        
        coords = self.parse_coordinates(response)  
        self.debug_print(f"InternVL3检测到 {len(coords)} 个元素")  
        
        if not coords:  
            return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs}  
        
        # 验证坐标并归一化  
        validated_coords = []  
        for coord in coords:  
            if not isinstance(coord, dict):  
                continue  
            
            # 获取像素坐标  
            x_pixel = float(coord.get('x', 0))  
            y_pixel = float(coord.get('y', 0))  
            
            # 归一化到 [0,1] 范围  
            x_norm = max(0.0, min(1.0, x_pixel / width))  
            y_norm = max(0.0, min(1.0, y_pixel / height))  
            
            validated_coords.append({  
                "label": coord.get('label', 'element'),  
                "x": x_norm,  
                "y": y_norm,  
                "x_pixel": x_pixel,  
                "y_pixel": y_pixel,  
                "confidence": float(coord.get('confidence', 0.7))  
            })  
            
            self.debug_print(f"归一化: [{x_pixel}, {y_pixel}] -> [{x_norm:.4f}, {y_norm:.4f}]")  
        
        if validated_coords:  
            validated_coords.sort(key=lambda x: x['confidence'], reverse=True)  
            best_match = validated_coords[0]  
            point = [best_match['x'], best_match['y']]  
            
            self.debug_print(f"最终选择: {best_match['label']} at ({point[0]:.3f}, {point[1]:.3f}), confidence={best_match['confidence']:.3f}")  
            
            return {  
                "result": "positive",  
                "bbox": None,  
                "point": point,  
                "raw_response": self.logs  
            }  
        
        return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs}  
  
    def ground_allow_negative(self, instruction, image):  
        """负样本检测(暂不支持)"""  
        raise NotImplementedError("InternVL3OnlyMethod暂不支持负样本检测")