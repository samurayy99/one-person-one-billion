"""
Business analysis utilities for Engin's AI Startup Universe.
This module contains functions for performing comparative analysis and scoring.
"""

from typing import Dict, Any
import logging
from data_utils import safe_float_conversion

logger = logging.getLogger(__name__)


def perform_head_to_head_analysis(startup1: Dict[str, Any], startup2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform detailed analysis between two startups.
    """
    try:
        def extract_metrics(startup):
            data_table = startup.get('data_table', {})
            tam = safe_float_conversion(data_table.get(
                'Total Accessable Market', '$0B'), 'B')
            investment = safe_float_conversion(
                data_table.get('Investment', '$0M'), 'M')
            pre_market = safe_float_conversion(
                data_table.get('Pre-Market Value', '$0M'), 'M')
            roi = pre_market / investment if investment > 0 else 0
            return {'tam': tam, 'investment': investment, 'pre_market': pre_market, 'roi': roi}

        metrics1 = extract_metrics(startup1)
        metrics2 = extract_metrics(startup2)

        def calculate_scores(m1, m2):
            scores = {'startup1': {}, 'startup2': {}}
            # TAM Score
            if m1['tam'] > m2['tam']:
                scores['startup1']['tam_score'] = 100
                scores['startup2']['tam_score'] = int(
                    (m2['tam'] / m1['tam']) * 100) if m1['tam'] > 0 else 0
            else:
                scores['startup2']['tam_score'] = 100
                scores['startup1']['tam_score'] = int(
                    (m1['tam'] / m2['tam']) * 100) if m2['tam'] > 0 else 0
            # Investment Efficiency
            if m1['investment'] < m2['investment'] and m1['investment'] > 0:
                scores['startup1']['investment_score'] = 100
                scores['startup2']['investment_score'] = int(
                    (m1['investment'] / m2['investment']) * 100)
            elif m2['investment'] > 0:
                scores['startup2']['investment_score'] = 100
                scores['startup1']['investment_score'] = int(
                    (m2['investment'] / m1['investment']) * 100) if m1['investment'] > 0 else 0
            else:
                scores['startup1']['investment_score'], scores['startup2']['investment_score'] = 50, 50
            # Pre-Market Value
            if m1['pre_market'] > m2['pre_market']:
                scores['startup1']['premarket_score'] = 100
                scores['startup2']['premarket_score'] = int(
                    (m2['pre_market'] / m1['pre_market']) * 100) if m1['pre_market'] > 0 else 0
            else:
                scores['startup2']['premarket_score'] = 100
                scores['startup1']['premarket_score'] = int(
                    (m1['pre_market'] / m2['pre_market']) * 100) if m2['pre_market'] > 0 else 0
            # ROI Potential
            if m1['roi'] > m2['roi']:
                scores['startup1']['roi_score'] = 100
                scores['startup2']['roi_score'] = int(
                    (m2['roi'] / m1['roi']) * 100) if m1['roi'] > 0 else 0
            else:
                scores['startup2']['roi_score'] = 100
                scores['startup1']['roi_score'] = int(
                    (m1['roi'] / m2['roi']) * 100) if m2['roi'] > 0 else 0

            for key in ['startup1', 'startup2']:
                scores[key]['total_score'] = (scores[key]['tam_score'] * 0.3 + scores[key]['investment_score']
                                              * 0.25 + scores[key]['premarket_score'] * 0.25 + scores[key]['roi_score'] * 0.2)
            return scores

        scores = calculate_scores(metrics1, metrics2)

        if scores['startup1']['total_score'] > scores['startup2']['total_score']:
            winner = startup1
            rationale = f"{startup1['name']} has a stronger overall profile with better market positioning and financial metrics."
        else:
            winner = startup2
            rationale = f"{startup2['name']} shows superior potential with more attractive investment metrics and market opportunity."

        return {'winner': winner, 'scores': scores, 'rationale': rationale, 'metrics': {'startup1': metrics1, 'startup2': metrics2}}

    except Exception as e:
        logger.error(f"Error in analysis calculation: {e}")
        return {'winner': startup1, 'scores': {'startup1': {}, 'startup2': {}}, 'rationale': "Analysis could not be completed due to data issues.", 'metrics': {}}
