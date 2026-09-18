"""
Image-based plant diagnosis using Groq vision models.

The app automatically checks which supported vision model is available
to the current Groq API key/project, so the model ID does not need to
be changed manually when Groq updates availability.
"""

import base64
import re
import time

from groq import Groq


# Preferred vision models, newest first.
PREFERRED_VISION_MODELS = (
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
)


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _mime_from_filename(filename: str) -> str:
    ext = (filename or "").lower().rsplit(".", 1)[-1]

    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")


def _get_accessible_vision_model(client: Groq) -> str:
    """
    Ask Groq which models are available to this API key/project,
    then select an accessible vision-capable model.
    """

    models = client.models.list()

    available_ids = {
        getattr(model, "id", "")
        for model in models.data
    }

    for model_id in PREFERRED_VISION_MODELS:
        if model_id in available_ids:
            return model_id

    raise RuntimeError(
        "No supported Groq vision model is available to this API key/project. "
        "The app checked: "
        + ", ".join(PREFERRED_VISION_MODELS)
        + ". Please enable/allow at least one vision model in your Groq project."
    )


def diagnose_image(
    api_key: str,
    image_bytes: bytes,
    filename: str,
    known_issues: list[str],
) -> dict:
    """
    Sends the uploaded photo to an accessible Groq vision model.

    Returns:
        {
            "matched_issue": str | None,
            "raw_response": str,
            "error": str | None,
            "model_used": str | None,
        }
    """

    try:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to the project-root .env file."
            )

        client = Groq(api_key=api_key)

        # Automatically find a vision model accessible to this key.
        vision_model = _get_accessible_vision_model(client)

        b64 = encode_image_bytes(image_bytes)
        mime = _mime_from_filename(filename)

        issue_list_str = ", ".join(known_issues)

        prompt = f"""You are a plant-care assistant looking at a photo of a plant.

Known issue categories this app can give detailed structured care plans for:

{issue_list_str}

Look at the photo and:

1. On the very first line, write:
   "MATCH: <issue name>"

   Use the EXACT name from the list above if the photo clearly resembles
   one of those issues.

   Otherwise write:
   "MATCH: none"

2. Then, in a few short paragraphs, describe:
   - what you observe in the photo,
   - your best guess about the plant problem,
   - general care/cure suggestions.

3. Keep the tone friendly and clear.

4. Do not claim certainty you don't have.
   Photos alone cannot always distinguish similar-looking issues.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}"
                        },
                    },
                ],
            }
        ]

        completion = None
        last_error = None

        # Retry temporary failures.
        for attempt in range(4):
            try:
                completion = client.chat.completions.create(
                    model=vision_model,
                    reasoning_format="hidden",
                    reasoning_effort="none",
                    messages=messages,
                    temperature=0.4,
                    max_completion_tokens=700,
                )
                break

            except Exception as exc:
                last_error = exc
                text = str(exc).lower()

                retryable = any(
                    token in text
                    for token in (
                        "429",
                        "rate limit",
                        "too many requests",
                        "500",
                        "502",
                        "503",
                        "504",
                        "timeout",
                        "timed out",
                        "temporarily",
                        "connection reset",
                        "connection error",
                    )
                )

                if not retryable or attempt == 3:
                    raise

                time.sleep(1.5 * (2 ** attempt))

        if completion is None:
            raise last_error or RuntimeError(
                "Vision model returned no response."
            )

        raw = completion.choices[0].message.content or ""

        # Remove accidental reasoning blocks.
        raw = re.sub(
            r"<think>.*?</think>",
            "",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )

        raw = re.sub(
            r"</?think(?:ing)?>",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()

        matched_issue = None

        first_line_match = re.match(
            r"\s*MATCH:\s*(.+)",
            raw.strip(),
            re.IGNORECASE,
        )

        if first_line_match:
            candidate = (
                first_line_match.group(1)
                .strip()
                .strip(".")
            )

            for issue in known_issues:
                if candidate.lower() == issue.lower():
                    matched_issue = issue
                    break

            # Remove MATCH line from displayed response.
            stripped = raw.strip()

            if "\n" in stripped:
                raw = stripped.split("\n", 1)[1].strip()
            else:
                raw = ""

        return {
            "matched_issue": matched_issue,
            "raw_response": raw,
            "error": None,
            "model_used": vision_model,
        }

    except Exception as e:
        return {
            "matched_issue": None,
            "raw_response": "",
            "error": str(e),
            "model_used": None,
        }