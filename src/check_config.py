import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. 定位 .env 文件
ENV_PATH = PROJECT_ROOT / ".env"

# 3. 加载环境变量
load_dotenv(ENV_PATH)

# 4. 读取配置
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model_name = os.getenv("DEEPSEEK_MODEL")

# 5. 安全检查：不输出真实密钥
print("===== DeepSeek 配置检查 =====")

print(".env 文件存在：", ENV_PATH.is_file())

print("API Key 已加载：", bool(api_key and api_key.strip()))

print("Base URL：", base_url)

print("模型名称：", model_name)
