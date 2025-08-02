"""
Data loading and processing utilities for Engin's AI Startup Universe.
This module handles all interactions with the raw data source (final_data.json)
and prepares it for the application.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import re

logger = logging.getLogger(__name__)


@st.cache_data
def load_startup_data() -> Dict[str, Any]:
    """
    Load startup data with robust error handling.
    Returns processed data with validation and error recovery.
    """
    try:
        with open('final_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'startup_blueprints' not in data:
            raise ValueError("Missing 'startup_blueprints' key in data")

        processed_startups = []
        for startup in data.get('startup_blueprints', []):
            processed_startup = validate_and_process_startup(startup)
            if processed_startup:
                processed_startups.append(processed_startup)

        data['startup_blueprints'] = processed_startups
        logger.info(f"Successfully loaded {len(processed_startups)} startups")

        return data

    except FileNotFoundError:
        logger.error("final_data.json not found")
        st.error(
            "📁 Data file not found. Please ensure final_data.json is in the project directory.")
        return {'startup_blueprints': []}

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        st.error("🔧 Data file format error. Please check the JSON structure.")
        return {'startup_blueprints': []}

    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        st.error(f"❌ Unexpected error: {str(e)}")
        return {'startup_blueprints': []}


def validate_and_process_startup(startup: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and process individual startup data with error recovery.
    """
    try:
        processed = {
            'id': startup.get('id', 0),
            'name': startup.get('name', 'Unknown Startup'),
            'description': startup.get('description', 'No description available'),
            'industry_focus': startup.get('industry_focus', 'Other'),
            'source_pages': startup.get('source_pages', []),
        }

        data_table = startup.get('data_table', {})
        processed['data_table'] = {
            'Total Accessable Market': safe_extract_financial_value(data_table.get('Total Accessable Market', '$0B'), 'B'),
            'Target Market': safe_extract_financial_value(data_table.get('Target Market', '$0B'), 'B'),
            'Investment': safe_extract_financial_value(data_table.get('Investment', '$0M'), 'M'),
            'Pre-Market Value': safe_extract_financial_value(data_table.get('Pre-Market Value', '$0M'), 'M')
        }

        canvas = startup.get('canvas', {})
        processed['canvas'] = {
            'Customer Segments': canvas.get('Customer Segments', 'Not specified'),
            'Problem': canvas.get('Problem', 'Not specified'),
            'Unique Value Proposition': canvas.get('Unique Value Proposition', 'Not specified'),
            'Proposed Solution': canvas.get('Proposed Solution', 'Not specified'),
            'Key Metrics': canvas.get('Key Metrics', 'Not specified'),
            'Channels': canvas.get('Channels', 'Not specified'),
            'Revenue Stream': canvas.get('Revenue Stream', 'Not specified'),
            'Cost Structure': canvas.get('Cost Structure', 'Not specified'),
            'Unfair Advantage': canvas.get('Unfair Advantage', 'Not specified'),
            'Key Risks & Mitigation': canvas.get('Key Risks & Mitigation', 'Not specified')
        }

        return processed

    except Exception as e:
        logger.warning(
            f"Error processing startup {startup.get('name', 'Unknown')}: {e}")
        return None


def safe_extract_financial_value(value_str: str, suffix: str) -> str:
    """
    Safely extract and validate financial values.
    """
    try:
        if not isinstance(value_str, str):
            return f"$0{suffix}"

        cleaned = value_str.replace('$', '').replace(',', '')

        if suffix in cleaned:
            numeric_part = cleaned.replace(suffix, '').strip()
            try:
                float(numeric_part)
                return f"${numeric_part}{suffix}"
            except ValueError:
                pass

        return f"$0{suffix}"

    except Exception:
        return f"$0{suffix}"


def safe_float_conversion(value_str: str, suffix: str) -> float:
    """
    Robustly convert financial string to float with comprehensive error handling.
    """
    try:
        if not value_str or value_str in ['N/A', 'n/a', 'NA', 'None', 'null', '']:
            return 0.0

        if not isinstance(value_str, str):
            try:
                value_str = str(value_str)
            except (TypeError, ValueError):
                logger.warning(
                    f"Could not convert {type(value_str)} to string: {value_str}")
                return 0.0

        cleaned = (value_str
                   .replace('$', '').replace('€', '').replace('£', '')
                   .replace(',', '').replace(' ', '').strip())

        if suffix and cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]

        if not cleaned or cleaned in ['-', 'TBD', 'TBA']:
            return 0.0

        try:
            numeric_value = float(cleaned)
            if numeric_value < 0:
                logger.warning(
                    f"Negative value detected: '{value_str}' -> {numeric_value}, returning 0.0")
                return 0.0
            if numeric_value > 1e12:
                logger.warning(
                    f"Unusually large value: '{value_str}' -> {numeric_value}")
            return numeric_value
        except (ValueError, OverflowError) as e:
            logger.warning(
                f"Numeric conversion failed for '{cleaned}' from '{value_str}': {e}")
            return 0.0

    except Exception as e:
        logger.error(
            f"Critical error in safe_float_conversion for '{value_str}': {e}")
        return 0.0


@st.cache_data
def create_analytics_dataframe(startups: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create analytics dataframe with safe data processing.
    """
    try:
        df_data = []
        for startup in startups:
            try:
                data_table = startup.get('data_table', {})
                tam_value = safe_float_conversion(
                    data_table.get('Total Accessable Market', '$0B'), 'B')
                investment_value = safe_float_conversion(
                    data_table.get('Investment', '$0M'), 'M')
                pre_market_value = safe_float_conversion(
                    data_table.get('Pre-Market Value', '$0M'), 'M')
                try:
                    roi_potential = (
                        pre_market_value / investment_value if investment_value > 0 else 0.0)
                    if roi_potential > 1000:
                        logger.warning(
                            f"Extremely high ROI detected for {startup.get('name', 'Unknown')}: {roi_potential}")
                        roi_potential = min(roi_potential, 1000)
                except (ZeroDivisionError, TypeError, ValueError):
                    roi_potential = 0.0

                df_data.append({
                    'id': startup.get('id', 0),
                    'name': startup.get('name', 'Unknown'),
                    'tam': tam_value,
                    'investment': investment_value,
                    'pre_market_value': pre_market_value,
                    'roi_potential': roi_potential,
                    'category': categorize_startup(startup)
                })
            except Exception as e:
                logger.warning(f"Error processing startup for analytics: {e}")
                continue

        df = pd.DataFrame(df_data)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        return df
    except Exception as e:
        logger.error(f"Error creating analytics dataframe: {e}")
        return pd.DataFrame()


def categorize_startup(startup: Dict[str, Any]) -> str:
    """
    Return the startup's industry category based on the industry_focus field from JSON.
    """
    try:
        # Get the industry_focus field directly from the startup data
        industry_focus = startup.get('industry_focus', '')

        # Map industry_focus to display categories with emojis
        industry_map = {
            'Al Tooling': '🤖 AI Tooling',
            'AdTech/Retail': '📱 AdTech/Retail',
            'SMB Marketing': '📈 SMB Marketing',
            'AgriTech': '🌾 AgriTech',
            'Autonomous Vehicles / Logistics': '🚗 Autonomous Vehicles / Logistics',
            'Data Analytics': '📊 Data Analytics',
            'DevOps': '⚙️ DevOps',
            'Telecom': '📡 Telecom',
            'HR Tech': '👥 HR Tech',
            'Healthcare': '🏥 Healthcare',
            'Energy': '⚡ Energy',
            'AI Architecture': '🏗️ AI Architecture',
            'Longevity': '🧬 Longevity',
            'Finance': '💰 Finance',
            'Travel/ Aging': '✈️ Travel/ Aging',
            'Home / DIY': '🏠 Home / DIY',
            'IT/HR Tech': '💻 IT/HR Tech',
            'Enterprise IT': '🏢 Enterprise IT',
            'Pharma/R&D': '💊 Pharma/R&D',
            'Fintech': '💳 Fintech',
            'Data Privacy': '🔒 Data Privacy',
            'Trade/ Legal': '⚖️ Trade/ Legal',
            'Infrastructure': '🏗️ Infrastructure',
            'FleetTech': '🚛 FleetTech',
            'Reg Tech': '📋 Reg Tech',
            'Cybersecurity': '🛡️ Cybersecurity',
            'Consumer IP': '🎬 Consumer IP',
            'Data Stack': '📚 Data Stack',
            'Defense': '🛡️ Defense',
            'Retail': '🛒 Retail',
            'Helpdesk': '🎧 Helpdesk',
            'Sales/CRM': '🎯 Sales/CRM',
            'Logistic': '📦 Logistic',
            'Supply Chain': '🔗 Supply Chain',
            'Legal Tech': '⚖️ Legal Tech',
            'Health Tech': '💊 Health Tech',
            'Industrial Al': '🏭 Industrial Al',
            'Enterprise Data': '🗄️ Enterprise Data',
            'Sales Ops': '📈 Sales Ops',
            'Transportation': '🚚 Transportation',
            'SaaS': '☁️ SaaS',
            'Al Stack': '🤖 Al Stack',
            'Gov Tech': '🏛️ Gov Tech',
            'Gaming': '🎮 Gaming'
        }

        # Return mapped category or the original industry_focus with a default emoji
        return industry_map.get(industry_focus, f'🌟 {industry_focus}')

    except Exception:
        return '🌟 Other'


def filter_startups_by_criteria(startups: List[Dict[str, Any]], industry: str = "All", investment_range: str = "All", tam_range: str = "All") -> List[Dict[str, Any]]:
    """
    Filter startups based on multiple criteria with error handling.
    """
    try:
        filtered = startups.copy()
        if industry != "All":
            filtered = [
                s for s in filtered if categorize_startup(s) == industry]
        if investment_range != "All":
            filtered = [s for s in filtered if filter_by_investment(
                s, investment_range)]
        if tam_range != "All":
            filtered = [s for s in filtered if filter_by_tam(s, tam_range)]
        return filtered
    except Exception as e:
        logger.error(f"Error filtering startups: {e}")
        return startups


def filter_by_investment(startup: Dict[str, Any], range_filter: str) -> bool:
    try:
        investment = safe_float_conversion(startup.get(
            'data_table', {}).get('Investment', '$0M'), 'M')
        if range_filter == "< $3M":
            return investment < 3
        elif range_filter == "$3M - $5M":
            return 3 <= investment <= 5
        elif range_filter == "> $5M":
            return investment > 5
        return True
    except Exception:
        return True


def filter_by_tam(startup: Dict[str, Any], range_filter: str) -> bool:
    try:
        tam = safe_float_conversion(startup.get('data_table', {}).get(
            'Total Accessable Market', '$0B'), 'B')
        if range_filter == "< $20B":
            return tam < 20
        elif range_filter == "$20B - $50B":
            return 20 <= tam <= 50
        elif range_filter == "> $50B":
            return tam > 50
        return True
    except Exception:
        return True


def calculate_universe_stats(startups: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate aggregated statistics for the startup universe.
    """
    try:
        total_tam = sum(safe_float_conversion(s.get('data_table', {}).get(
            'Total Accessable Market', '$0B'), 'B') for s in startups)
        total_investment = sum(safe_float_conversion(
            s.get('data_table', {}).get('Investment', '$0M'), 'M') for s in startups)
        total_pre_market = sum(safe_float_conversion(
            s.get('data_table', {}).get('Pre-Market Value', '$0M'), 'M') for s in startups)
        return {
            'total_startups': len(startups),
            'total_tam': total_tam,
            'total_investment': total_investment,
            'total_pre_market': total_pre_market,
            'avg_roi': total_pre_market / total_investment if total_investment > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error calculating universe stats: {e}")
        return {'total_startups': 0, 'total_tam': 0, 'total_investment': 0, 'total_pre_market': 0, 'avg_roi': 0}


def get_industry_categories() -> List[str]:
    """
    Get list of available industry categories.
    """
    return ["All", "🎯 Sales & CRM", "⚖️ Legal & Compliance", "🏥 Healthcare", "💰 Finance", "🛒 Retail & Commerce", "🏭 Industry & Logistics", "🤖 AI & Automation", "🎮 Gaming & Entertainment", "🌱 Climate & Sustainability", "🚀 Space & Advanced Tech", "🌟 Other"]


def validate_startup_data_integrity(startups: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validate data integrity and return validation results.
    """
    errors = []
    try:
        if not startups:
            errors.append("No startup data available")
            return False, errors
        for i, startup in enumerate(startups):
            startup_id = startup.get('id', f"unknown_{i}")
            if not startup.get('name'):
                errors.append(f"Startup {startup_id}: Missing name")
            if not startup.get('description'):
                errors.append(f"Startup {startup_id}: Missing description")
            if not startup.get('data_table', {}):
                errors.append(f"Startup {startup_id}: Missing data_table")
            if not startup.get('canvas', {}):
                errors.append(f"Startup {startup_id}: Missing canvas")
        return len(errors) == 0, errors
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, errors
