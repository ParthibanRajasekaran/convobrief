"""Summarizer prompts and LLM interface.

Contains the system and user prompts for meeting summarization as specified.
"""

from typing import Any

from insightsvc.logging import get_logger
from insightsvc.models.base import ModelMetadata, SummarizerModel

logger = get_logger(__name__)

# System prompt for meeting analysis
SYSTEM_PROMPT = """You are a careful meeting analyst.
Tasks:
1) Produce a concise summary (<= {max_words} words) of the conversation.
2) List decisions, action items (owner, due date if stated), blockers, and open questions.
3) Capture disagreements and tone shifts. If sarcasm was detected, quote the line and note the real intent.
4) Do NOT invent facts. Only use the transcript. Prefer exact phrases with timestamps.
5) Keep speaker labels (S1..S5). Use UTC timestamps in mm:ss format.
Output JSON with keys: summary, decisions[], action_items[], disagreements[], risks[], open_questions[].
"""

# User prompt template
USER_PROMPT_TEMPLATE = """CONTEXT:
- Speakers: {speakers_json}
- Transcript (speaker-attributed words & utterances): {transcript_json}
- Sarcasm & mood signals: {mood_json}

INSTRUCTIONS:
- Be neutral and faithful. No speculation.
- When uncertain, say "uncertain" and include the evidence span.
- Respect {max_words} word budget for `summary`.
"""

# Sarcasm arbiter prompt
SARCASM_ARBITER_PROMPT = """You are detecting sarcasm in dialogue without losing context.
Input includes the last 3 utterances and the current utterance with speaker labels and timestamps.
Answer JSON: { "is_sarcastic": boolean, "rationale": string <= 25 words }.
Rules:
- Sarcasm often pairs positive wording with negative situation or mismatched prosody.
- Do not call jokes or exaggeration sarcasm unless there is implicit criticism or mock praise.
"""

# Mood arbiter prompt
MOOD_ARBITER_PROMPT = """Given (a) text sentiment scores, (b) audio valence/arousal, (c) sarcasm flag,
return JSON: { "valence": 0..1, "label": "negative|neutral|positive", "confidence": 0..1 }.
Prefer audio valence when text and audio disagree and sarcasm=true.
"""


class LLMSummarizer(SummarizerModel):
    """LLM-based meeting summarizer.

    TODO: Implement using HF transformers with Mistral or Llama.
    """

    def __init__(self, model_name: str, device: str):
        """Initialize summarizer.

        Args:
            model_name: HF model name (e.g., mistralai/Mistral-7B-Instruct-v0.3).
            device: Device for inference.
        """
        self.model_name = model_name
        self.device_name = device

        logger.info("Initializing summarizer", model_name=model_name, device=device)

        # TODO: Load model
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        #     device_map="auto"
        # )

        logger.info("Summarizer loaded", model_name=model_name)

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return ModelMetadata(
            name=self.model_name,
            version=None,
            config={},
            device=self.device_name,
        )

    def to(self, device: str) -> "LLMSummarizer":
        """Move model to device."""
        self.device_name = device
        # TODO: Move model
        return self

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Stop sequences.

        Returns:
            Generated text.
        """
        logger.info(
            "Generating summary",
            prompt_length=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # TODO: Implement generation
        # Format prompt with model-specific template (Mistral/Llama)
        # Run generation with stopping criteria
        # Extract and validate JSON output

        raise NotImplementedError("Summarizer generation not yet implemented")

    def summarize_meeting(
        self,
        speakers: list[dict[str, Any]],
        transcript: str,
        mood_signals: str,
        max_words: int = 250,
    ) -> dict[str, Any]:
        """Summarize meeting using prompts.

        Args:
            speakers: Speaker metadata list.
            transcript: JSON string of transcript.
            mood_signals: JSON string of mood analysis.
            max_words: Maximum words for summary.

        Returns:
            Dictionary with summary, decisions, action_items, etc.
        """
        import json

        # Build prompt
        system_prompt = SYSTEM_PROMPT.format(max_words=max_words)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            speakers_json=json.dumps(speakers),
            transcript_json=transcript,
            mood_json=mood_signals,
            max_words=max_words,
        )

        # Format for model (example for Mistral)
        full_prompt = f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"

        # Generate
        output = self.generate(
            full_prompt,
            max_tokens=1024,
            temperature=0.2,
            stop_sequences=["[/INST]", "</s>"],
        )

        # Parse JSON output
        try:
            result = json.loads(output)
            return result
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM output", output=output)
            # Return minimal structure
            return {
                "summary": "Failed to generate summary",
                "decisions": [],
                "action_items": [],
                "disagreements": [],
                "risks": [],
                "open_questions": [],
            }
