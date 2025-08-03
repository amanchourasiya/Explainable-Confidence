# generate.py

import google.generativeai as genai
import re

#
import config

class ExplainableCodeGenerator:
    def __init__(self, tuned_model_name):
        print("--- Initializing Generator with Fine-Tuned Gemini Model ---")
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(tuned_model_name)
        print("Generator ready.")

    def _parse_output(self, text):
        """A simple parser for the model's structured output."""
        code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
        explanation_match = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
        
        code = code_match.group(1).strip() if code_match else "Parsing Error: No <code> tag found."
        explanation = explanation_match.group(1).strip() if explanation_match else "Parsing Error: No <explanation> tag found."
        
        return code, explanation

    def generate(self, prompt):
        """Generates a single response for a given prompt using the fine-tuned model."""
        
        #
        full_prompt = f"<prompt>{prompt}</prompt>"
        
        response = self.model.generate_content(full_prompt)
        
        #
        code, explanation = self._parse_output(response.text)
        
        return {"code": code, "explanation": explanation}

if __name__ == "__main__":
    #
    # IMPORTANT: Replace this with the actual name of your successfully tuned model
    #
    TUNED_MODEL = f"tunedModels/{config.TUNED_MODEL_DISPLAY_NAME}" 
    
    generator = ExplainableCodeGenerator(tuned_model_name=TUNED_MODEL)
    
    test_prompt = "Write a Python function to check if a string is a palindrome."
    result = generator.generate(test_prompt)
    
    print("\n--- Generated Code ---")
    print(result["code"])
    print("\n--- Explanation ---")
    print(result["explanation"])
