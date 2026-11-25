
from unittest.mock import patch, MagicMock

from groq import RateLimitError

from rag.generator import AnswerGenerator


generator = AnswerGenerator()

# Create a mock HTTP response compatible with Groq's SDK.
mock_response = MagicMock()
mock_response.status_code = 429
mock_response.headers = {}
mock_response.request = MagicMock()

# Simulate Groq returning a rate-limit error.
simulated_error = RateLimitError(
    message="Simulated rate limit",
    response=mock_response,
    body={
        "error": {
            "message": "Simulated rate limit",
            "code": "rate_limit_exceeded"
        }
    }
)

print("Simulating Groq rate limit...")

with patch.object(
    generator.client.chat.completions,
    "create",
    side_effect=simulated_error
):

    response = generator._create_completion(
        model=generator.model,
        messages=[
            {
                "role": "user",
                "content": "Explain an n8n workflow briefly."
            }
        ],
        temperature=0.2
    )

    print("\nFallback response:")
    print(response.choices[0].message.content)

    print("\nAutomatic fallback test completed.")
