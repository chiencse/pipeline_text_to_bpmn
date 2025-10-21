import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyABhnUhDweDUlT2zCYziR_F67XCYhQn7bU")

for m in genai.list_models():
    caps = getattr(m, "supported_generation_methods", []) or getattr(m, "generation_methods", [])
    if "generateContent" in caps:
        print(m.name)
