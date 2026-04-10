import fitz  # PyMuPDF，用于读取 PDF
from openai import OpenAI
import json

# ==========================================
# [需要你填充] 1. 配置 DeepSeek API
# ==========================================
# 请将下方字符串替换为你在 DeepSeek 平台申请的实际 API Key
DEEPSEEK_API_KEY = "sk-32ee9ee867124dc79915b3673efd55f8"

# 初始化 OpenAI 客户端，但指向 DeepSeek 的服务器
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com" # DeepSeek的官方API接口地址
)

# ==========================================
# [需要你填充] 2. 配置文件路径
# ==========================================
# 填入你下载的 PDF 文件的绝对或相对路径
PDF_FILE_PATH =  "C:/Users/邹童阳/Desktop/task1/task1.pdf"

def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取纯文本
    """
    print("正在读取 PDF 文件内容...")
    text = ""
    try:
        # 打开 PDF 文件
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"读取 PDF 时发生错误: {e}")
        return None

def extract_entities_with_deepseek(text):
    """
    调用 DeepSeek API 提取医学实体
    """
    print("正在调用 DeepSeek 大模型进行实体提取（这可能需要几十秒）...")
    
    # 构建提示词 (Prompt)，明确我们需要的结构化实体
    prompt = f"""
    你是一个专业的医学信息提取助手。请阅读以下病例报告（Case Report）的文本内容，并从中准确提取关键的医学实体。
    
    请将提取的结果以严格的 JSON 格式输出。请包含以下字段（如果在文本中找不到对应信息，请将该字段的值填为"未提及"）：
    
    {{
        "patient_info": "患者基本信息（如年龄、性别等）",
        "chief_complaint_and_symptoms": "主诉及主要临床症状",
        "medical_history": "既往病史（Past medical history）",
        "physical_examination": "体格检查关键发现",
        "laboratory_and_imaging": "重要的实验室检查及影像学检查结果",
        "diagnosis": "明确的诊断结果",
        "treatment_and_intervention": "治疗方案、用药或手术干预",
        "outcome": "患者预后、随访或治疗结果"
    }}

    以下是病例的文本内容：
    \"\"\"
    {text}
    \"\"\"
    """

    try:
        # 调用大模型
        response = client.chat.completions.create(
            model="deepseek-chat", # 使用 deepseek 的通用对话模型
            messages=[
                {"role": "system", "content": "你是一个严谨的医疗AI助手，你只能输出结构化的 JSON 数据，不要包含任何额外的解释性文字。"},
                {"role": "user", "content": prompt}
            ],
            # 强制要求输出 JSON 格式
            response_format={"type": "json_object"},
            temperature=0.1 # 调低温度以保证输出结果的确定性和严谨性
        )
        
        # 解析返回的 JSON 字符串
        result_json = response.choices[0].message.content
        return json.loads(result_json)
        
    except Exception as e:
        print(f"调用 API 或解析结果时发生错误: {e}")
        return None

def main():
    # 1. 提取文本
    case_text = extract_text_from_pdf(PDF_FILE_PATH)
    if not case_text:
         return
    
    # 提示：如果 PDF 特别长（超过几万字），可能需要考虑截断或分段。
    # 但单篇 Case Report 通常只有几页，完全在 DeepSeek 的上下文窗口（通常支持 32K+）范围内。
    
    # 2. 提取实体
    entities = extract_entities_with_deepseek(case_text)
    
    # 3. 打印结果
    if entities:
        print("\n================ 提取的关键医学实体 ================\n")
        # 格式化输出 JSON
        print(json.dumps(entities, indent=4, ensure_ascii=False))
        
        # （可选）将结果保存到本地文件
        with open("extracted_entities.json", "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=4, ensure_ascii=False)
        print("\n==================================================")
        print("提取结果已保存至 extracted_entities.json")

if __name__ == "__main__":
    main()