import re

class LexicalGuardrail:
    """
    Failsafe Layer: Prevents adversarial prompts, injection attacks, 
    or Out-Of-Distribution (OOD) transcripts from skewing the semantic fusion.
    """
    def __init__(self):
        # Basic regex rules to block code injections, SQL injections, or anomalous lengths
        self.forbidden_patterns = [
            r"(<script>.*?</script>)",
            r"(SELECT.*FROM.*)",
            r"(DROP\s+TABLE.*)",
            r"([{}<>]+)", # Anomalous braces/brackets often found in code
            r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)" # URLs
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.forbidden_patterns]

    def sanitize(self, text: str) -> str:
        """
        Validates the transcript. Returns a safe fallback string if malicious 
        intent or severe OOD is detected, preserving the system's stability.
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Length bounds check (Prevents Buffer Overflows or massive computation delays on CPU)
        if len(text.strip()) == 0 or len(text) > 500:
            return "[NEUTRAL_FALLBACK_TEXT]"
            
        # 2. Pattern check
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                print(f"[SECURITY ALERT] Malicious pattern detected in text. Dropping semantic load.")
                return "[NEUTRAL_FALLBACK_TEXT]"
                
        return text.strip()

# Global instance
guardrail = LexicalGuardrail()
