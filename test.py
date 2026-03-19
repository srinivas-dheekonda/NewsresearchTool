from dotenv import load_dotenv
import os
load_dotenv()
print(f"Is API Key loaded?: {bool(os.getenv('GROQ_API_KEY'))}")