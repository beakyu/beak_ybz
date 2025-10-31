import os  
import re  
import json  
import base64  
import requests  
from PIL import Image  
import io  
  
class GeminiOnlyMethod:  
    """单独使用 Gemini 进行检测"""  
  
    def __init__(self, planner=None, grounder=None, configs=None):  
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")  
        if not self.openrouter_api_key:  
            raise ValueError("请设置 OPENROUTER_API_KEY 环境变量")  
  
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
        """将PIL图像编码为base64"""  
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
        buffer = io.BytesIO()  
        image.save(buffer, format='JPEG', quality=85)  
        return base64.b64encode(buffer.getvalue()).decode('utf-8')  
  
    def call_gemini(self, prompt, image):  
        """调用Gemini API"""  
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
        """主入口: 仅使用 Gemini 进行检测"""  
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
  
        self.debug_print(f"Gemini单独检测: {instruction}")  
  
        # 使用与组合模型相同的 prompt  
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
        if not response:  
            return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs}  
  
        coords = self.parse_coordinates(response)  
        self.debug_print(f"Gemini检测到 {len(coords)} 个元素")  
  
        if not coords:  
            return {"result": "negative", "bbox": None, "point": None, "raw_response": self.logs} 
        
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