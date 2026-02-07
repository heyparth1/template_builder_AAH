# config.py - Configuration and Schema Definitions for Template Builder

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_MODEL = "gpt-4-turbo-preview"

# Chunking Configuration
CHUNK_SIZES = {
    'small': 1024,
    'medium': 2048,
    'max': 4096
}

DEFAULT_CHUNK_SIZE = 'max'

# Trigger phrases that indicate client-specific content (for paragraph detection)
CLIENT_TRIGGER_PHRASES = [
    "you ", "your ", "you're", "yours",
    "we recommend", "we suggest", "we advise",
    "based on your", "given your", "considering your",
    "your portfolio", "your investments", "your goals",
    "your risk", "your circumstances", "your situation",
    "during our meeting", "when we met", "we discussed",
    "I recommend", "I suggest", "I advise",
    "for you", "to you", "with you",
    "your account", "your fund", "your pension",
    "you have", "you are", "you will", "you should", "you could", "you may",
    "we have agreed", "you agreed", "you decided",
    "your family", "your spouse", "your children",
    "tailored to", "designed for", "specific to",
    "in your case", "for your needs"
]

# Dynamic Content Schema
DYNAMIC_CONTENT_SCHEMA = {
    "DATE": {
        "description": "Specific dates",
        "placeholder": "{{DATE}}",
        "patterns": [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
            r"\b(?:Q[1-4]|H[1-2])\s+\d{4}\b",
            r"\b\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
        ]
    },
    "CURRENCY": {
        "description": "Currency amounts",
        "placeholder": "{{CURRENCY_AMOUNT}}",
        "patterns": [
            r"[$£€¥]\s*[\d,]+(?:\.\d{1,2})?(?:\s*[KMBkmb](?:illion)?)?",
            r"(?:USD|GBP|EUR|JPY|CHF|AUD|CAD)\s*[\d,]+(?:\.\d{1,2})?",
            r"[\d,]+(?:\.\d{1,2})?\s*(?:USD|GBP|EUR|pounds?|dollars?|euros?)",
            r"£[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand|[KMBkmb]))?",
            r"\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand|[KMBkmb]))?",
            r"€[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand|[KMBkmb]))?",
            r"[\d,]+(?:\.\d{1,2})?\s*(?:p\.a\.|per annum|annually|/month|per month)"
        ]
    },
    "CUSTOMER_NAME": {
        "description": "Customer/client names",
        "placeholder": "{{CUSTOMER_NAME}}",
        "patterns": [
            r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
        ]
    },
    "PERCENTAGE": {
        "description": "Specific percentages",
        "placeholder": "{{PERCENTAGE}}",
        "patterns": [
            r"\d+\.?\d*\s*%",
            r"\d+\.?\d*\s*percent",
            r"\d+\.?\d*\s*per\s*cent"
        ]
    },
    "CLIENT_PARAGRAPH": {
        "description": "ENTIRE paragraphs about this specific client",
        "placeholder": "{{CLIENT_PARAGRAPH}}",
        "patterns": []
    },
    "CLIENT_GOAL": {
        "description": "Client-specific financial goals, objectives, targets",
        "placeholder": "{{CLIENT_GOAL}}",
        "patterns": []
    },
    "CLIENT_INSIGHT": {
        "description": "Observations about THIS specific client",
        "placeholder": "{{CLIENT_INSIGHT}}",
        "patterns": []
    },
    "CLIENT_ADVICE": {
        "description": "Recommendations tailored to THIS client",
        "placeholder": "{{CLIENT_ADVICE}}",
        "patterns": []
    },
    "INVESTMENT_DETAIL": {
        "description": "Specific investment holdings, fund names, allocations",
        "placeholder": "{{INVESTMENT_DETAIL}}",
        "patterns": []
    },
    "ALLOCATION": {
        "description": "Financial allocations, budget items with amounts",
        "placeholder": "{{ALLOCATION}}",
        "patterns": []
    },
    "RISK_STATEMENT": {
        "description": "Risk warnings specific to THIS client's situation",
        "placeholder": "{{RISK_STATEMENT}}",
        "patterns": []
    },
    "MEETING_SUMMARY": {
        "description": "Meeting notes, review summaries about client",
        "placeholder": "{{MEETING_SUMMARY}}",
        "patterns": []
    },
    "ADDRESS": {
        "description": "Customer addresses",
        "placeholder": "{{ADDRESS}}",
        "patterns": []
    },
    "PHONE": {
        "description": "Phone numbers",
        "placeholder": "{{PHONE_NUMBER}}",
        "patterns": [
            r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
        ]
    },
    "EMAIL": {
        "description": "Email addresses",
        "placeholder": "{{EMAIL}}",
        "patterns": [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ]
    },
    "ACCOUNT_NUMBER": {
        "description": "Account numbers",
        "placeholder": "{{ACCOUNT_NUMBER}}",
        "patterns": []
    }
}

# Static Content Rules
STATIC_CONTENT_RULES = [
    "Section headings and titles",
    "Questions in Q&A format", 
    "Generic statements applicable to all clients",
    "Regulatory disclaimers",
    "Firm information and policies",
    "Table headers and labels"
]

# === MAIN PROMPT: Focus on PARAGRAPHS ===
CLASSIFICATION_PROMPT_TEMPLATE = '''You are creating a reusable document template. Your PRIMARY task is to find and replace CLIENT-SPECIFIC PARAGRAPHS (not just individual values).

DOCUMENT TEXT:
{text_chunk}

=== CRITICAL: READ EVERY PARAGRAPH ===

For EACH paragraph in the document, ask yourself:
"Does this paragraph talk about THIS SPECIFIC CLIENT, or is it generic advice that applies to everyone?"

TRIGGER WORDS that indicate CLIENT-SPECIFIC content (MUST REPLACE the whole paragraph):
- "you", "your", "you're", "yours" (when referring to the client)
- "we recommend", "we suggest", "we advise", "I recommend"
- "based on your", "given your", "considering your"
- "during our meeting", "when we met", "we discussed"
- "your portfolio", "your investments", "your goals", "your risk"
- Any paragraph describing what THIS client wants, has, or should do

=== WHAT TO REPLACE ===

1. ENTIRE PARAGRAPHS containing client-specific information:
   - Paragraphs about client's goals, objectives, or targets
   - Paragraphs describing client's current situation
   - Paragraphs with recommendations for THIS client
   - Paragraphs summarizing meetings or discussions with client
   - Paragraphs about risks specific to THIS client's investments
   - Bullet points or numbered lists describing client's allocations
   
2. Individual values (always replace):
   - ALL currency amounts (£X, $X, €X)
   - ALL percentages (X%)
   - ALL dates
   - Names, addresses, phone numbers, emails

=== WHAT TO KEEP (DO NOT REPLACE) ===

- Headings and section titles (e.g., "Business Expansion", "Risk Assessment")
- Questions (e.g., "What are your goals?")  
- Generic statements that apply to ALL clients
- Standard disclaimers and warnings
- Table headers

=== EXAMPLES ===

REPLACE THIS PARAGRAPH:
"Increase annual business revenue to £500,000 within the next 3 years through strategic growth initiatives."
→ {{{{CLIENT_GOAL}}}} (this describes THIS client's specific goal)

REPLACE THIS PARAGRAPH:
"Your investments include overseas assets, and any fluctuations in exchange rates can impact the value of your investments."
→ {{{{CLIENT_PARAGRAPH}}}} (says "your investments" = client-specific)

REPLACE THIS PARAGRAPH:
"During our meeting, we discussed your personal, family and work circumstances, the economic environment, your attitude towards risk."
→ {{{{MEETING_SUMMARY}}}} (describes meeting with THIS client)

KEEP THIS (generic):
"The value of investments can rise and fall, and you may get back less than you invested."
→ DO NOT REPLACE (standard disclaimer for everyone)

KEEP THIS (heading):
"Business Expansion"
→ DO NOT REPLACE (this is a section heading)

=== OUTPUT FORMAT ===

Return ONLY valid JSON. For paragraphs, return the COMPLETE paragraph text:

{{
  "replacements": [
    {{
      "original": "the complete paragraph or value to replace - copy EXACTLY",
      "placeholder": "{{{{CATEGORY}}}}",
      "category": "CLIENT_PARAGRAPH|CLIENT_GOAL|CLIENT_INSIGHT|CLIENT_ADVICE|MEETING_SUMMARY|RISK_STATEMENT|INVESTMENT_DETAIL|ALLOCATION|DATE|CURRENCY|PERCENTAGE|CUSTOMER_NAME|PHONE|EMAIL|ACCOUNT_NUMBER",
      "confidence": 0.9,
      "reason": "contains trigger word X / describes client's Y"
    }}
  ]
}}

IMPORTANT: Do NOT be lazy! Scan EVERY paragraph. If it contains "you/your" referring to the client, it needs to be replaced.

If no dynamic content found: {{"replacements": []}}'''

# Rate Limiting Configuration
MAX_TOKENS_PER_MINUTE = 90000
MAX_REQUESTS_PER_MINUTE = 500
RATE_LIMIT_BUFFER = 0.9

# File upload limits
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {'.docx'}

# Output configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
