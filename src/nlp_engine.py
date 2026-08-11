import os
import math
from datetime import datetime, timezone
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

from enum import Enum

class CatalystType(Enum):
    EARNINGS = "EARNINGS"
    FDA_APPROVAL = "FDA_APPROVAL"
    CONTRACT_WIN = "CONTRACT_WIN"
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    OFFERING_DILUTION = "OFFERING_DILUTION"
    RUMOR = "RUMOR"
    OTHER = "OTHER"

# Load FinBERT model globally to avoid reloading on each call
FINBERT_MODEL = "ProsusAI/finbert"
try:
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    # Ensure evaluation mode
    model.eval()
except Exception as e:
    logger.error(f"Failed to load FinBERT: {e}")
    tokenizer = None
    model = None

def get_catalyst_category(headline: str) -> str:
    """Uses LLM to extract primary catalyst category for highly impactful headlines."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found. Defaulting to OTHER.")
        return CatalystType.OTHER.value
        
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=openai_api_key)
        parser = StrOutputParser()
        
        prompt = PromptTemplate(
            template="""You are a quantitative financial analyst. 
Given the following news headline, categorize the primary catalyst.
Headline: {headline}

Must be exactly one of: EARNINGS, FDA_APPROVAL, CONTRACT_WIN, MANAGEMENT_CHANGE, OFFERING_DILUTION, RUMOR, OTHER. Return only the string.""",
            input_variables=["headline"],
        )
        
        chain = prompt | llm | parser
        result = chain.invoke({"headline": headline}).strip()
        
        if result in [e.value for e in CatalystType]:
            return result
        return CatalystType.OTHER.value
    except Exception as e:
        logger.error(f"LLM extraction failed for '{headline}': {e}")
        return CatalystType.OTHER.value

def score_news_headlines(headlines: list[dict]) -> pd.DataFrame:
    """
    Scores headlines using FinBERT, applies time-decay, and extracts catalyst for significant news.
    Input: [{'ticker': 'XYZ', 'headline': '...', 'timestamp': datetime_obj}]
    """
    if not headlines:
        return pd.DataFrame()
        
    if not tokenizer or not model:
        logger.error("FinBERT model is not loaded. Returning empty dataframe.")
        return pd.DataFrame(headlines)
        
    texts = [item['headline'] for item in headlines]
    
    # Tokenize in batch
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        
    # FinBERT class order: [positive, negative, neutral]
    pos_probs = probs[:, 0].numpy()
    neg_probs = probs[:, 1].numpy()
    
    # Compute raw sentiment S = P(pos) - P(neg)
    raw_sentiments = pos_probs - neg_probs
    
    current_time = datetime.now(timezone.utc)
    decayed_sentiments = []
    catalysts = []
    
    for i, item in enumerate(headlines):
        S = raw_sentiments[i]
        pub_time = item['timestamp']
        
        # Ensure timezone-aware
        if pub_time.tzinfo is None:
            pub_time = pub_time.replace(tzinfo=timezone.utc)
            
        elapsed_timedelta = current_time - pub_time
        t_hours = elapsed_timedelta.total_seconds() / 3600.0
        
        # Prevent negative time (if pub_time is somehow in future)
        t_hours = max(0, t_hours)
        
        # Weekend time bridge: if news spans over the weekend, subtract 48 hours
        # Check if pub_time is before the weekend and current_time is after the weekend
        if pub_time.weekday() in [4, 5, 6] and current_time.weekday() in [0, 1, 2]:
            t_hours = max(0, t_hours - 48.0)
        
        # Decay formula: S_decayed = S * exp(- (ln(2)/3.0) * t)
        half_life = 3.0
        decay_factor = math.exp(-(math.log(2) / half_life) * t_hours)
        S_decayed = S * decay_factor
        
        decayed_sentiments.append(S_decayed)
        
        # Trigger LLM extraction if highly impactful
        if abs(S_decayed) > 0.4:
            cat = get_catalyst_category(item['headline'])
            catalysts.append(cat)
        else:
            catalysts.append(CatalystType.OTHER.value)
            
    df = pd.DataFrame(headlines)
    df['raw_sentiment'] = raw_sentiments
    df['decayed_sentiment'] = decayed_sentiments
    df['catalyst_category'] = catalysts
    
    return df
