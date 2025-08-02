"""
AI Service module for external API calls and AI-powered features.
Production-grade AI integration with robust error handling and caching.
"""

import streamlit as st
import openai
import random
import time
from typing import Dict, List, Any, Optional
from functools import wraps, lru_cache
import logging
import json

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e

                    # Calculate delay with exponential backoff + jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter

                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {total_delay:.2f}s")
                    time.sleep(total_delay)

            # This should never be reached, but just in case
            raise last_exception if last_exception else Exception(
                "Unknown error in retry decorator")

        return wrapper
    return decorator


class AIService:
    """
    Centralized AI service for all external API calls.
    """

    def __init__(self):
        """
        Initialize AI service with API keys from Streamlit secrets.
        """
        self.api_key = self._get_api_key()
        self.client = None

        if self.api_key:
            try:
                openai.api_key = self.api_key
                self.client = openai
                logger.info("AI Service initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing AI service: {e}")

    def _get_api_key(self) -> Optional[str]:
        """
        Safely retrieve API key from Streamlit secrets.
        """
        try:
            # Try to get OpenAI API key from secrets
            if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
                return st.secrets['openai_api_key']

            logger.warning("OpenAI API key not found in secrets")
            return None

        except Exception as e:
            logger.error(f"Error retrieving API key: {e}")
            return None

    def is_available(self) -> bool:
        """
        Check if AI service is available.
        """
        return self.api_key is not None and self.client is not None

    @retry_with_exponential_backoff(
        max_retries=3,
        base_delay=1.0,
        exceptions=(openai.RateLimitError, openai.APITimeoutError,
                    openai.APIConnectionError)
    )
    def _make_api_call(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", **kwargs) -> str:
        """
        Make API call to OpenAI with retry logic and enhanced error handling.

        Args:
            messages: List of message dictionaries for the API call
            model: Model to use (default: gpt-3.5-turbo)
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response text

        Raises:
            Various OpenAI exceptions, handled by retry decorator
        """
        try:
            if not self.is_available():
                raise ValueError(
                    "AI service not available - no API key configured")

            # Set default parameters
            api_params = {
                'model': model,
                'messages': messages,
                'max_tokens': kwargs.get('max_tokens', 200),
                'temperature': kwargs.get('temperature', 0.7),
                'top_p': kwargs.get('top_p', 0.9),
                'timeout': kwargs.get('timeout', 30)  # 30 second timeout
            }

            logger.info(
                f"Making API call to {model} with {len(messages)} messages")

            response = self.client.chat.completions.create(**api_params)

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from OpenAI API")

            content = response.choices[0].message.content.strip()

            if not content:
                raise ValueError("Empty content in OpenAI response")

            logger.info(
                f"Successful API call, received {len(content)} characters")
            return content

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            raise ValueError(
                "API authentication failed - please check your API key")
        except openai.PermissionDeniedError as e:
            logger.error(f"OpenAI permission denied: {e}")
            raise ValueError("API access denied - insufficient permissions")
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit: {e}")
            raise  # Let retry decorator handle this
        except openai.APITimeoutError as e:
            logger.warning(f"OpenAI API timeout: {e}")
            raise  # Let retry decorator handle this
        except openai.APIConnectionError as e:
            logger.warning(f"OpenAI connection error: {e}")
            raise  # Let retry decorator handle this
        except openai.BadRequestError as e:
            logger.error(f"OpenAI bad request: {e}")
            raise ValueError(f"Invalid request to OpenAI API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            raise ValueError(f"Unexpected API error: {e}")

    def _create_cache_key(self, startup: Dict[str, Any], prompt_type: str = "pitch") -> str:
        """
        Create a hashable cache key for startup data.

        Args:
            startup: Startup data dictionary
            prompt_type: Type of prompt (e.g., 'pitch', 'risk', 'recommendation')

        Returns:
            String cache key
        """
        try:
            # Use key components that define the prompt
            key_components = [
                str(startup.get('id', 0)),
                startup.get('name', ''),
                prompt_type,
                # First 100 chars of description
                str(hash(startup.get('description', '')[:100]))
            ]

            return "|".join(key_components)

        except Exception as e:
            logger.warning(f"Error creating cache key: {e}")
            return f"{startup.get('id', 0)}|{prompt_type}|default"

    @lru_cache(maxsize=128)  # Cache up to 128 AI responses in memory
    def _cached_generate_pitch(self, cache_key: str, name: str, problem: str, solution: str, market: str, investment: str) -> str:
        """
        Generate elevator pitch with LRU caching.

        Args:
            cache_key: Unique cache key
            name: Startup name
            problem: Problem statement
            solution: Solution description
            market: Market opportunity
            investment: Investment needed

        Returns:
            Generated elevator pitch
        """
        try:
            prompt = f"""
            Create a compelling, professional elevator pitch for this startup opportunity:

            Company: {name or 'AI Startup'}
            Market Problem: {problem or 'Business efficiency challenges'}
            Value Proposition: {solution or 'AI-powered solution'}
            Market Opportunity: {market or 'Significant market'}
            Investment Required: {investment or 'Funding needed'}

            Requirements:
            - 2-3 sentences maximum
            - Professional, investor-focused tone
            - Highlight scalability and market potential
            - Avoid overly technical jargon
            - Focus on business value and ROI
            """

            messages = [
                {"role": "system", "content": "You are an expert business strategist and pitch consultant. Create professional, compelling elevator pitches that resonate with investors and entrepreneurs."},
                {"role": "user", "content": prompt}
            ]

            pitch = self._make_api_call(
                messages=messages,
                model="gpt-4o",
                max_tokens=200,
                temperature=0.6,
                top_p=0.9
            )

            logger.info(
                f"Generated cached pitch for key: {cache_key[:50]}... ({len(pitch)} chars)")
            return pitch

        except Exception as e:
            logger.error(f"Error in cached pitch generation: {e}")
            raise

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def generate_elevator_pitch(_self, startup: Dict[str, Any]) -> str:
        """
        Generate AI-powered elevator pitch for a startup with robust caching and fallbacks.
        """
        try:
            startup_id = startup.get('id', 0)

            if not _self.is_available():
                fallback_pitch = _self._fallback_elevator_pitch(startup)
                logger.info(
                    f"AI service unavailable, returning fallback pitch for startup {startup_id}")
                return fallback_pitch

            # Create cache key for LRU cache
            lru_cache_key = _self._create_cache_key(startup, "pitch")

            # Extract data for caching
            canvas = startup.get('canvas', {})
            data_table = startup.get('data_table', {})

            name = startup.get('name', 'AI Startup')
            problem = canvas.get('Problem', 'Business efficiency challenges')
            solution = canvas.get(
                'Unique Value Proposition', 'AI-powered solution')
            market = data_table.get(
                'Total Accessable Market', 'Significant market')
            investment = data_table.get('Investment', 'Funding needed')

            try:
                # Use LRU cached method for AI generation
                pitch = _self._cached_generate_pitch(
                    cache_key=lru_cache_key,
                    name=name,
                    problem=problem,
                    solution=solution,
                    market=market,
                    investment=investment
                )

                # Validate pitch quality
                if len(pitch) < 50:
                    logger.warning(
                        f"Generated pitch too short ({len(pitch)} chars), using fallback")
                    fallback_pitch = _self._fallback_elevator_pitch(startup)
                    return fallback_pitch

                logger.info(
                    f"Successfully generated/retrieved cached AI pitch for startup {startup_id} ({len(pitch)} chars)")
                return pitch

            except ValueError as ve:
                # These are our custom errors from _make_api_call
                logger.error(f"API configuration error: {ve}")
                fallback_pitch = _self._fallback_elevator_pitch(startup)
                st.warning(
                    "AI pitch generation temporarily unavailable. Using demo pitch.")
                return fallback_pitch + " (Demo mode - API configuration needed)"

            except Exception as api_error:
                logger.error(f"Error in pitch generation: {api_error}")
                fallback_pitch = _self._fallback_elevator_pitch(startup)
                st.error(
                    "Pitch generation temporarily unavailable. Please try again later.")
                return fallback_pitch + " (Generated with fallback due to technical issue)"

        except Exception as e:
            logger.error(f"Error generating elevator pitch: {e}")
            fallback_pitch = "Pitch generation temporarily unavailable. Please try again later."
            return fallback_pitch

    def _fallback_elevator_pitch(self, startup: Dict[str, Any]) -> str:
        """
        Fallback elevator pitch generation without AI.
        """
        try:
            canvas = startup.get('canvas', {})
            data_table = startup.get('data_table', {})

            problem = canvas.get(
                'Problem', 'businesses face efficiency challenges')[:100]
            solution = canvas.get('Unique Value Proposition',
                                  'AI-powered automation')[:100]
            tam = data_table.get('Total Accessable Market',
                                 'billion-dollar market')
            name = startup.get('name', 'this startup')

            pitches = [
                f"Imagine solving {problem.lower()} with {solution.lower()} - that's exactly what {name} delivers in the {tam} opportunity space. We're not just building software, we're creating the future of autonomous business operations.",

                f"While competitors struggle with {problem.lower()}, {name} transforms this challenge with {solution.lower()}. In a {tam} market, we're positioned to capture massive value through AI-native innovation.",

                f"What if {problem.lower()} could be completely eliminated? {name} makes this reality with {solution.lower()}, targeting the {tam} market with unprecedented efficiency and scale."
            ]

            return random.choice(pitches)

        except Exception as e:
            logger.error(f"Error in fallback pitch generation: {e}")
            return f"🚀 {startup.get('name', 'This startup')} is revolutionizing business automation with AI-native solutions, targeting a massive market opportunity for one-person, billion-dollar success!"

    @st.cache_data(ttl=3600)
    def get_startup_recommendations(_self,
                                    startups: List[Dict[str, Any]],
                                    interests: List[str],
                                    skills: List[str],
                                    risk_tolerance: str,
                                    investment_capacity: str) -> List[tuple]:
        """
        Get AI-powered startup recommendations based on user preferences.
        """
        try:
            if not _self.is_available():
                return _self._fallback_recommendations(startups, interests, skills, risk_tolerance, investment_capacity)

            # Score each startup
            scored_startups = []

            for startup in startups:
                score = _self._calculate_recommendation_score(
                    startup, interests, skills, risk_tolerance, investment_capacity
                )
                scored_startups.append((startup, score))

            # Sort by score and return top recommendations
            return sorted(scored_startups, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return _self._fallback_recommendations(startups, interests, skills, risk_tolerance, investment_capacity)

    def _calculate_recommendation_score(self,
                                        startup: Dict[str, Any],
                                        interests: List[str],
                                        skills: List[str],
                                        risk_tolerance: str,
                                        investment_capacity: str) -> float:
        """
        Calculate recommendation score for a startup.
        """
        try:
            score = 0.0
            desc = startup.get('description', '').lower()
            canvas = startup.get('canvas', {})
            data_table = startup.get('data_table', {})

            # Interest matching (40% weight)
            interest_score = 0
            for interest in interests:
                if interest.lower().replace("/", " ").replace("-", " ") in desc:
                    interest_score += 1
            score += (interest_score / max(len(interests), 1)) * 40

            # Skill relevance (30% weight)
            skill_keywords = {
                'Technical/Programming': ['ai', 'automation', 'platform', 'api', 'cloud'],
                'Sales': ['sales', 'crm', 'customer', 'revenue'],
                'Marketing': ['marketing', 'campaign', 'engagement', 'brand'],
                'Design': ['ui', 'ux', 'design', 'interface'],
                'Business Strategy': ['strategy', 'business', 'model', 'planning'],
                'Data Analysis': ['data', 'analytics', 'insights', 'intelligence'],
                'Product Management': ['product', 'management', 'workflow', 'process'],
                'Operations': ['operations', 'logistics', 'supply', 'automation']
            }

            skill_score = 0
            for skill in skills:
                keywords = skill_keywords.get(skill, [])
                if any(keyword in desc for keyword in keywords):
                    skill_score += 1
            score += (skill_score / max(len(skills), 1)) * 30

            # Investment capacity matching (20% weight)
            try:
                investment_needed = float(data_table.get(
                    'Investment', '$0M').replace('$', '').replace('M', ''))
                if investment_capacity == "< $2M" and investment_needed < 2:
                    score += 20
                elif investment_capacity == "$2M - $4M" and 2 <= investment_needed <= 4:
                    score += 20
                elif investment_capacity == "> $4M" and investment_needed > 4:
                    score += 20
                elif investment_capacity == "> $4M" and investment_needed <= 4:
                    score += 15  # Partial match
            except:
                pass

            # Risk tolerance (10% weight)
            if 'freemium' in desc.lower() and risk_tolerance == "Conservative (Low Risk)":
                score += 10
            elif risk_tolerance == "Aggressive (High Risk)":
                score += 5
            elif risk_tolerance == "Moderate":
                score += 7

            return min(score, 100)

        except Exception as e:
            logger.error(f"Error calculating recommendation score: {e}")
            return random.uniform(20, 80)

    def _fallback_recommendations(self,
                                  startups: List[Dict[str, Any]],
                                  interests: List[str],
                                  skills: List[str],
                                  risk_tolerance: str,
                                  investment_capacity: str) -> List[tuple]:
        """
        Fallback recommendation system without AI.
        """
        try:
            scored_startups = []

            for startup in startups:
                score = self._calculate_recommendation_score(
                    startup, interests, skills, risk_tolerance, investment_capacity
                )
                # Add some randomness for variety
                score += random.uniform(-10, 10)
                scored_startups.append((startup, max(0, min(100, score))))

            return sorted(scored_startups, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return [(startup, random.uniform(50, 90)) for startup in startups]

    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def analyze_startup_risks(_self, startup: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze startup risks using AI or rule-based approach.
        """
        try:
            if not _self.is_available():
                return _self._fallback_risk_analysis(startup)

            canvas = startup.get('canvas', {})
            risks_text = canvas.get('Key Risks & Mitigation', '')

            # For now, use rule-based analysis
            # In production, this could use AI for more sophisticated analysis
            return _self._fallback_risk_analysis(startup)

        except Exception as e:
            logger.error(f"Error analyzing risks: {e}")
            return _self._fallback_risk_analysis(startup)

    def _fallback_risk_analysis(self, startup: Dict[str, Any]) -> Dict[str, float]:
        """
        Rule-based risk analysis fallback.
        """
        try:
            canvas = startup.get('canvas', {})
            desc = startup.get('description', '').lower()
            data_table = startup.get('data_table', {})

            risks = {
                'Technical Risk': 50,
                'Market Risk': 50,
                'Competition Risk': 50,
                'Regulatory Risk': 50,
                'Financial Risk': 50,
                'Operational Risk': 50
            }

            # Adjust based on keywords and characteristics
            if 'ai' in desc or 'automation' in desc:
                risks['Technical Risk'] += 15

            if 'healthcare' in desc or 'finance' in desc:
                risks['Regulatory Risk'] += 20

            if 'freemium' in desc:
                risks['Financial Risk'] -= 10

            try:
                investment = float(data_table.get(
                    'Investment', '$0M').replace('$', '').replace('M', ''))
                if investment > 5:
                    risks['Financial Risk'] += 15
                elif investment < 2:
                    risks['Financial Risk'] -= 10
            except:
                pass

            # Add some randomness
            for risk_type in risks:
                risks[risk_type] += random.randint(-15, 15)
                risks[risk_type] = max(20, min(90, risks[risk_type]))

            return risks

        except Exception as e:
            logger.error(f"Error in fallback risk analysis: {e}")
            return {
                'Technical Risk': 60,
                'Market Risk': 55,
                'Competition Risk': 65,
                'Regulatory Risk': 45,
                'Financial Risk': 50,
                'Operational Risk': 55
            }

    @st.cache_data(ttl=3600)
    def generate_battle_commentary(_self, startup1: Dict[str, Any], startup2: Dict[str, Any], winner: Dict[str, Any]) -> str:
        """
        Generate AI commentary for startup battles.
        """
        try:
            if not _self.is_available():
                return _self._fallback_battle_commentary(startup1, startup2, winner)

            # For now, use fallback
            return _self._fallback_battle_commentary(startup1, startup2, winner)

        except Exception as e:
            logger.error(f"Error generating battle commentary: {e}")
            return _self._fallback_battle_commentary(startup1, startup2, winner)

    def _fallback_battle_commentary(self, startup1: Dict[str, Any], startup2: Dict[str, Any], winner: Dict[str, Any]) -> str:
        """
        Fallback battle commentary generation.
        """
        try:
            winner_name = winner.get('name', 'Unknown')
            loser_name = (startup1 if startup1.get('id') != winner.get(
                'id') else startup2).get('name', 'Unknown')

            commentaries = [
                f"🏆 In a fierce battle of innovation, {winner_name} emerges victorious! Their superior market positioning and investment efficiency give them the edge over {loser_name}.",

                f"⚡ What a showdown! {winner_name} demonstrates why they're built for billion-dollar success, outperforming {loser_name} in key metrics that matter most to investors.",

                f"🎯 The verdict is in! {winner_name} takes the crown with their compelling value proposition and market advantage, proving they have what it takes to outcompete {loser_name}.",

                f"🚀 In this epic clash of AI-native startups, {winner_name} shows their dominance! Their strategic approach and execution potential clearly surpass {loser_name}'s offering."
            ]

            return random.choice(commentaries)

        except Exception as e:
            logger.error(f"Error in fallback battle commentary: {e}")
            return f"🏆 {winner.get('name', 'The winner')} claims victory in this startup battle!"

    def ask_sparring_partner(self, chat_history: List[Dict[str, str]]) -> str:
        """
        🧠 Elite AI Sparring Partner - Specialized Startup Expert

        Difference from regular ChatGPT:
        - Passionate about this specific startup idea
        - Knows all details from Business Model Canvas  
        - Acts as Co-Founder, Investor and Mentor combined
        - Provides concrete, actionable advice for this exact idea
        """
        if not self.is_available():
            return "🚫 **Elite AI Sparring Partner Unavailable**\n\nPremium expertise requires API access. Configure your OpenAI key to unlock world-class startup insights."

        try:
            # Enhanced API call with specialized configuration for startup expertise
            response = self._make_api_call(
                messages=chat_history,
                model="gpt-4.1",  # Premium model for premium expertise
                max_tokens=1500,  # More detailed, comprehensive responses
                temperature=0.8,  # Slightly higher for creative business insights
                top_p=0.95  # High quality tokens for professional discourse
            )

            logger.info(
                f"AI Sparring Partner delivered {len(response)} chars of specialized expertise")
            return response

        except Exception as e:
            logger.error(f"Error in ask_sparring_partner: {e}")
            return f"🔥 **Elite AI Sparring Partner Temporarily Offline**\n\n{e}\n\n*Even the best systems need a moment. Try again for world-class startup insights.*"


# Global AI service instance
ai_service = AIService()
