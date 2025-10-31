
import os  
import json  
import base64  
from io import BytesIO  
from PIL import Image  
import requests  
import time  
import re 
from PIL import ImageDraw 
  
  
def convert_pil_image_to_base64(image):  
    """Convert PIL Image to base64 string."""  
    buffered = BytesIO()  
    image.save(buffered, format="PNG")  
    img_str = base64.b64encode(buffered.getvalue()).decode()  
    return img_str  
  
  
class QwenUITarsComboMethod:  
    def __init__(self, uitars_model="bytedance/ui-tars-1.5-7b",  
                 qwen_model="qwen/qwen2.5-vl-72b-instruct",  
                 api_base="https://openrouter.ai/api/v1/chat/completions"):  
        """  
        双层组合模型: Qwen初检 + UI-TARS修正  
          
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

    
    def plot_annotated_circle(self, image, point, radius=15, is_correct=None, alpha=100):  
        """在图片上标注预测点 - 使用半透明实心圆  
        
        Args:  
            image: PIL Image  
            point: [x_pixel, y_pixel] 像素坐标  
            radius: 圆的半径(像素),默认15  
            is_correct: True(蓝色)/False(红色)/None(默认红色)  
            alpha: 透明度(0-255),默认100(较透明,避免遮挡)  
        """  

        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))  
        draw = ImageDraw.Draw(overlay)  
        
        if point is not None:  
            x, y = point  
              
            if is_correct:  
                color = (0, 0, 255, alpha)  # 蓝色 + 透明度  
            else:  
                color = (255, 0, 0, alpha)  # 红色 + 透明度  

            draw.ellipse(  
                (x - radius, y - radius, x + radius, y + radius),  
                fill=color,   
                outline=None  
            )  
         
        annotated_image = image.convert('RGBA')  
        annotated_image = Image.alpha_composite(annotated_image, overlay)  
        return annotated_image.convert('RGB') 
      
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
        """解析像素坐标"""  
        if not response:  
            return None  
        
        try:  
            response = re.sub(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\)', r'[\1, \2]', response)  
            
            if '"coordinate"' in response:  
                open_brackets = response.count('[')  
                close_brackets = response.count(']')  
                open_braces = response.count('{')  
                close_braces = response.count('}')  
                
                if open_brackets > close_brackets:  
                    response += ']' * (open_brackets - close_brackets)  
                if open_braces > close_braces:  
                    response += '}' * (open_braces - close_braces)  
            
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
  
            match = re.search(r'"coordinate"\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)', response)  
            if match:  
                x_pixel = float(match.group(1))  
                y_pixel = float(match.group(2))  
                self.debug_print(f"提取像素坐标: [{x_pixel}, {y_pixel}]")  
                return [x_pixel, y_pixel]  

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
        """第一层: Qwen初检"""  
        width, height = image.size  
        
        prompt = f"""You are a GUI automation assistant. The screen resolution is {width}x{height}.  
    
    Task: {instruction}  
    
    Use the computer_use tool to click on the target element.  
    
    Available tool:  
    - computer_use(action: str, coordinate: list[int, int])  
    - action: "left_click"  
    - coordinate: [x, y] pixel position where x∈[0,{width}], y∈[0,{height}]  
    
    Output the tool call in JSON:  
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
    
    Return ONLY the JSON, no explanation.""" 
        
        response = self.call_openrouter_api(self.qwen_model, prompt, image)  
        pixel_point = self.parse_pixel_coordinates_raw(response)  
        
        return pixel_point, response 
      
    def refine_with_uitars(self, instruction, image, initial_pixel_point, initial_response, is_correct=False):  
        """第二层: 修正"""  
        self.debug_print("=== Second layer: Refinement ===")  
        width, height = image.size  
        
        if initial_pixel_point:  
              
            annotated_image = self.plot_annotated_circle(  
                image,  
                initial_pixel_point,  
                radius=15,    
                is_correct=is_correct,  
                alpha=100    
            )  
        else:  
            annotated_image = image  
        
        x_pixel, y_pixel = initial_pixel_point if initial_pixel_point else (width//2, height//2)  
        
         
        prompt = f"""You are verifying a GUI detection result. A red circle marks the previous detection at [{x_pixel}, {y_pixel}].  
    
    ## Task  
    {instruction}  
    
    ## Your Job  
    1. Check if the red circle is on the correct target  
    2. If YES → output SAME coordinates [{x_pixel}, {y_pixel}]  
    3. If NO → find the correct element and output its center  
    
    ## Output Format  
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
    
    ## Action Space  
    click(coordinate=[x, y]) where x∈[0,{width}], y∈[0,{height}]"""   
        
        response = self.call_openrouter_api(self.uitars_model, prompt, annotated_image)  
        self.debug_print(f"Refinement response: {response}")  
        refined_point = self.parse_pixel_coordinates_raw(response)  
        return refined_point, response  
    
    def ground_only_positive(self, instruction, image):  
        """主入口: 执行双层grounding"""  
        self.logs = []  
        
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
        
        width, height = image.size  

        qwen_pixel_point, qwen_response = self.ground_with_qwen_initial(instruction, image)  
        
        if qwen_pixel_point is None:  
            return {  
                "result": "negative",  
                "point": None,  
                "bbox": None,  
                "raw_response": {"qwen_initial": qwen_response, "logs": self.logs}  
            }  
        
        time.sleep(1)  
        
        final_pixel_point, uitars_response = self.refine_with_uitars(  
            instruction, image, qwen_pixel_point, qwen_response, is_correct=False  
        )  

        if final_pixel_point is None:  
            self.debug_print("UI-TARS修正失败,回退到Qwen初检结果")  
            final_pixel_point = qwen_pixel_point  
        
        final_normalized_point = self.normalize_pixel_coordinates(final_pixel_point, width, height)  
        
        return {  
            "result": "positive" if final_normalized_point else "negative",  
            "point": final_normalized_point,  
            "bbox": None,  
            "raw_response": {  
                "qwen_initial": qwen_response,  
                "uitars_refined": uitars_response,  
                "logs": self.logs  
            }  
        }
      
    def ground_allow_negative(self, instruction, image):  
        """支持负样本的grounding"""  
        return self.ground_only_positive(instruction, image)
