# search_project/gemini_client.py
import os
import re
import google.generativeai as genai
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'

genai_model_instance: Optional[genai.GenerativeModel] = None

try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        logger.warning(
            "GOOGLE_API_KEY environment variable not set. Gemini features will be disabled. "
            "Set this key to enable AI-powered suggestions and summaries."
        )
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        genai_model_instance = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        logger.info(f"Gemini client initialized successfully with model: {GEMINI_MODEL_NAME}.")
except Exception as e:
    logger.error(f"Error initializing Gemini client with model '{GEMINI_MODEL_NAME}': {e}. Gemini features disabled.")
    genai_model_instance = None

def get_autocomplete_suggestions(partial_query: str, mode: str = "GENERAL", max_suggestions: int = 5) -> List[str]:
    if not genai_model_instance:
        logger.debug("Gemini model not available for autocomplete. Returning empty list.")
        return []

    if not partial_query or len(partial_query) < 2:
        return []

    try:
        if mode.upper() == "CODE":
            prompt = f"""You are an AI assistant that provides ultra-fast, relevant autocomplete suggestions for a CODE search engine.
User's partial code-related query: "{partial_query}"
Generate {max_suggestions} concise autocomplete suggestions. Each suggestion should be a direct continuation or a very closely related term.
Focus on: function names, variable types, error messages, library names, programming concepts, or language keywords.
Prioritize shorter suggestions. List one suggestion per line. No extra text or explanation.

Example 1:
User's partial code-related query: "python list comprehen"
Suggestions:
python list comprehension if else
python list comprehension multiple conditions
python list comprehension syntax

Example 2:
User's partial code-related query: "javascript TypeE"
Suggestions:
javascript TypeError undefined
javascript TypeError null
javascript TypeError cannot read property

Partial Query: "{partial_query}"
Suggestions:
"""
        else:
            prompt = f"""You are an AI assistant that provides ultra-fast, relevant autocomplete suggestions for a GENERAL-PURPOSE search engine.
User's partial query: "{partial_query}"
Generate {max_suggestions} concise autocomplete suggestions. Each suggestion should be a direct continuation or a very closely related term.
List one suggestion per line. No extra text or explanation.

Example 1:
User's partial query: "benefits of exercise for"
Suggestions:
benefits of exercise for mental health
benefits of exercise for weight loss
benefits of exercise for seniors

Example 2:
User's partial query: "how to learn a new lang"
Suggestions:
how to learn a new language fast
how to learn a new language effectively
how to learn a new language for free

Partial Query: "{partial_query}"
Suggestions:
"""
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=100,
            temperature=0.05
        )

        logger.debug(f"Sending autocomplete prompt to Gemini (mode: {mode}): '{prompt[:150]}...'")
        response = genai_model_instance.generate_content(prompt, generation_config=generation_config)

        suggestions: List[str] = []
        if response and response.text:
            raw_lines = response.text.split('\n')
            for line in raw_lines:
                clean_line = line.strip().replace("- ", "").replace("* ", "")
                if clean_line and len(clean_line) > 1 and clean_line.lower() not in ["suggestions:", "example suggestions:"]:
                    suggestions.append(clean_line)
            logger.debug(f"Gemini autocomplete raw response: '{response.text[:150] if response.text else 'Empty'}' -> Parsed: {suggestions}")
        else:
            logger.warning(f"Gemini autocomplete returned no text for query: '{partial_query}'")

        return suggestions[:max_suggestions]

    except Exception as e:
        logger.error(f"Error during Gemini autocomplete API call for '{partial_query}': {e}")
        return []

def get_ai_summary_or_fix(content: str, query: str, doc_type_str: str, language: Optional[str] = None) -> Optional[str]:
    if not genai_model_instance:
        logger.debug("Gemini model not available for AI summary/fix.")
        return None

    max_chars_for_content = 4000
    clean_content = re.sub(r'\n\s*\n', '\n', content).strip()
    if len(clean_content) > max_chars_for_content:
        clean_content = clean_content[:max_chars_for_content] + "\n... (content truncated for AI processing)"
        logger.debug(f"Content for AI assist truncated to {max_chars_for_content} characters.")

    prompt = ""
    if doc_type_str.upper() == "CODE":
        lang_context = f"The detected/provided programming language is: {language}." if language and language.lower() != "unknown" else "The programming language is not specified or is unknown."

        is_error_content = any(err_kw in clean_content.lower() for err_kw in ["error", "exception", "traceback", "panic:", "failed to"]) or \
                           re.search(r'\bat\s+[\w$.]+\([\w.]+:\d+\)', clean_content)

        if is_error_content:
            prompt = f"""User query: "{query}"
Context: The user found the following code-related error log or diagnostic message.
{lang_context}
```text
{clean_content}
```
Task:
Concise Explanation: In 1-2 sentences, explain the most likely root cause of this error.
Actionable Suggestion: Provide one primary, practical suggestion for how to fix or debug this issue. If the error is too generic, suggest how the user might gather more specific diagnostic information.
Output Format:
Explanation: <Your explanation>
Suggestion: <Your suggestion or debugging tip>
"""
        else:
            prompt = f"""User query: "{query}"
Context: The user found the following code snippet.
{lang_context}
```text
{clean_content}
```
Task:
Brief Explanation: In 1-2 sentences, explain the primary purpose or functionality of this code snippet.
Relevance (Optional): If the user's query hints at a specific problem or goal, briefly (1 sentence) relate the code to that or suggest how it might be used/improved in that context if obvious.
Output Format:
Explanation: <Your explanation of the code's purpose>
"""
    elif doc_type_str.upper() == "GENERAL":
        prompt = f"""User query: "{query}"
Context: The user found the following text snippet.
--- TEXT BEGIN ---
{clean_content}
--- TEXT END ---
Task: Provide a very concise (2-3 sentences) summary of this text. Focus on the aspects most relevant to the user's query. If the text is very short, the summary can be shorter.
Output Format:
Summary: <Your summary>
"""
    else:
        logger.warning(f"Unsupported document type '{doc_type_str}' for AI assist.")
        return None

    generation_config = genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=350,
        temperature=0.25
    )

    try:
        logger.debug(f"Sending AI assist prompt to Gemini (type: {doc_type_str}, lang: {language}, query: {query}) - Content snippet: '{clean_content[:100]}...'")
        response = genai_model_instance.generate_content(prompt, generation_config=generation_config)

        ai_response_text = response.text.strip() if response and response.text else None
        if ai_response_text:
            logger.debug(f"Gemini AI assist raw response: '{ai_response_text[:150]}...'")
        else:
            logger.warning(f"Gemini AI assist returned no text for query '{query}', type '{doc_type_str}'.")
        return ai_response_text
    except Exception as e:
        logger.error(f"Error during Gemini AI summary/fix API call (query: '{query}', type: '{doc_type_str}'): {e}")
        return None

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("SKIPPING Gemini client tests: GOOGLE_API_KEY environment variable is not set.")
    else:
        print("--- Testing Gemini Client (GOOGLE_API_KEY is set) ---")

        print("\n[Test 1] Autocomplete (CODE mode):")
        code_suggestions = get_autocomplete_suggestions("python string forma", mode="CODE")
        if code_suggestions:
            for i, s in enumerate(code_suggestions): print(f"  {i+1}. {s}")
        else:
            print("  No suggestions or error occurred.")

        print("\n[Test 2] Autocomplete (GENERAL mode):")
        general_suggestions = get_autocomplete_suggestions("history of artificial intelligen", mode="GENERAL")
        if general_suggestions:
            for i, s in enumerate(general_suggestions): print(f"  {i+1}. {s}")
        else:
            print("  No suggestions or error occurred.")

        print("\n[Test 3] AI Fix Suggestion (CODE ERROR):")
        sample_error = "FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'\n  File \"reader.py\", line 15, in read_data"
        fix = get_ai_summary_or_fix(sample_error, "python filenotfounderror", "CODE", "python")
        print(f"  AI Response:\n{fix if fix else '  No response or error.'}")

        print("\n[Test 4] AI Code Snippet Explanation:")
        sample_code = "def greet(name):\n    return f\"Hello, {name}!\""
        explanation = get_ai_summary_or_fix(sample_code, "python greeting function", "CODE", "python")
        print(f"  AI Response:\n{explanation if explanation else '  No response or error.'}")

        print("\n[Test 5] AI Summary (GENERAL text):")
        sample_text = "The Montreal Protocol is an international treaty designed to protect the ozone layer by phasing out the production of numerous substances that are responsible for ozone depletion. It was agreed on 16 September 1987, and entered into force on 1 January 1989."
        summary = get_ai_summary_or_fix(sample_text, "what is montreal protocol", "GENERAL")
        print(f"  AI Response:\n{summary if summary else '  No response or error.'}")

        print("\n[Test 6] AI Task Suggestion (CODE):")
        sample_code = "def process_data(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result"
        task = get_ai_summary_or_fix(sample_code, "add task to optimize processing", "CODE", "python")
        print(f"  AI Response:\n{task if task else '  No response or error.'}")

        print("\n--- Gemini Client Tests Finished ---")