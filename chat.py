from google import genai
import streamlit as st
from langchain_core.prompts import PromptTemplate
from google.genai import types
# ===== LẤY SECRETS =====
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]


custom_prompt_template = """
Yêu cầu cụ thể là tổng hợp thông tin trong các đoạn Context để trả lời câu hỏi. 

1. Nếu câu trả lời không có trong Context hoặc bạn không chắc chắn, hãy trả lời:
"Tôi không có đủ thông tin để trả lời câu hỏi này. Vui lòng cung cấp thêm thông tin liên quan đến câu hỏi."

2. Không suy đoán và bịa đặt nội dung ngoài.

3. Chỉ trả lời thông tin theo Context tìm được.

4. Context được chia cách bởi chuỗi "SEPARATED".

5. Chỉ sử dụng History khi người dùng hỏi về câu hỏi trước đó:

History: {history_global}

Context: {context}

Question: {question}

Câu trả lời:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["history_global", "context", "question"],
    )
    return prompt


class GeminiBot:
    def __init__(self):
        self.client = None
        self._setup()

    def _setup(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def response(self, user_input):
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=user_input,
            config=types.GenerateContentConfig(
                temperature=0.05,
                top_p=1,
                top_k=1,
                max_output_tokens=1000,
            )
        )
        return response.text


gemini_bot = GeminiBot()
