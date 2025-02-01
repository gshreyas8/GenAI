import os
import json
import pytest
from dotenv import load_dotenv
import openai

# Load .env once, so OPENAI_API_KEY (etc.) is available.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class OpenAIConnectionTester:
    """Tests basic connectivity to OpenAI's API."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Store API key and model; set openai.api_key for requests.
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key  # If using the official openai library.

    def check_connection(self) -> dict:
        """
        Make a simple ChatCompletion request; return a dict with success, message, and response.
        """
        result = {
            "success": False,
            "message": "Initial state",
            "error": None,
            "model": self.model,
            "response": None,
        }

        if not self.api_key:
            result["message"] = "No API key provided"
            return result

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": "Respond with 'API operational'",
                    }
                ],
                max_tokens=20,
            )
            content = response["choices"][0]["message"]["content"]
            usage = response["usage"]

            result.update(
                {
                    "success": True,
                    "message": "API connection successful",
                    "response": {
                        "content": content,
                        "model": response["model"],
                        "usage": {
                            "completion_tokens": usage["completion_tokens"],
                            "prompt_tokens": usage["prompt_tokens"],
                            "total_tokens": usage["total_tokens"],
                        },
                    },
                }
            )
        except Exception as e:
            result["message"] = "API connection failed"
            result["error"] = str(e)

        return result


@pytest.mark.api
def test_openai_connection():
    """
    Confirm OPENAI_API_KEY is set and a basic request to the model
    returns 'API operational'.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    tester = OpenAIConnectionTester(api_key=api_key, model="gpt-3.5-turbo")

    result = tester.check_connection()
    assert result[
        "success"
    ], f"API connection failed: {result.get('message')} - {result.get('error')}"

    print("\nAPI Response:")
    print(json.dumps(result, indent=2))

    # Check that the response content includes 'operational'
    assert (
        "operational" in result["response"]["content"].lower()
    ), f"Unexpected response content: {result['response']['content']}"
