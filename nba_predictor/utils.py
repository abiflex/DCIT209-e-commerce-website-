"""
Utility Functions
Helper functions for formatting and displaying predictions
"""

from typing import List
from .predictor import PredictionResult


def format_prediction(result: PredictionResult, detailed: bool = False) -> str:
    """
    Format prediction result as string
    
    Args:
        result: PredictionResult object
        detailed: Whether to include detailed statistics
        
    Returns:
        Formatted string representation
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  {result.away_team} @ {result.home_team}")
    lines.append(f"{'='*70}")
    lines.append(f"  Win Probability:")
    lines.append(f"    {result.home_team:15s} {result.home_win_prob:5.1f}%")
    lines.append(f"    {result.away_team:15s} {result.away_win_prob:5.1f}%")
    lines.append(f"")
    lines.append(f"  Predicted Score:")
    lines.append(f"    {result.home_team:15s} {result.home_score_mean:5.1f}")
    lines.append(f"    {result.away_team:15s} {result.away_score_mean:5.1f}")
    lines.append(f"")
    lines.append(f"  Spread:  {result.spread:+.1f} ({result.home_team})")
    lines.append(f"  Total:   {result.total:.1f}")
    
    if detailed:
        lines.append(f"")
        lines.append(f"  95% Confidence Interval (Spread):")
        lines.append(f"    [{result.confidence_interval_95[0]:+.1f}, {result.confidence_interval_95[1]:+.1f}]")
        lines.append(f"")
        lines.append(f"  Score Standard Deviations:")
        lines.append(f"    {result.home_team:15s} ±{result.home_score_std:.1f}")
        lines.append(f"    {result.away_team:15s} ±{result.away_score_std:.1f}")
        lines.append(f"")
        lines.append(f"  Simulations: {result.simulations:,}")
    
    lines.append(f"{'='*70}\n")
    
    return '\n'.join(lines)


def format_multiple_predictions(results: List[PredictionResult], 
                                detailed: bool = False) -> str:
    """Format multiple predictions"""
    output = []
    output.append("\n" + "="*70)
    output.append(f"  NBA GAME PREDICTIONS - {len(results)} Games")
    output.append("="*70)
    
    for result in results:
        output.append(format_prediction(result, detailed))
    
    return '\n'.join(output)


def export_predictions(results: List[PredictionResult], 
                      format: str = 'csv',
                      filepath: str = None) -> None:
    """
    Export predictions to file
    
    Args:
        results: List of PredictionResult objects
        format: Export format ('csv' or 'json')
        filepath: Output file path (auto-generated if None)
    """
    from .data_manager import DataManager
    from datetime import datetime
    
    if filepath is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"predictions_{timestamp}.{format}"
    
    if format.lower() == 'csv':
        DataManager.export_predictions_csv(results, filepath)
    elif format.lower() == 'json':
        DataManager.export_predictions_json(results, filepath)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")
    
    print(f"✓ Predictions exported to: {filepath}")


def create_comparison_table(results: List[PredictionResult]) -> str:
    """Create a compact comparison table for multiple games"""
    lines = []
    lines.append("\n" + "="*100)
    lines.append(f"{'MATCHUP':<30} {'WIN %':<15} {'SCORE':<20} {'SPREAD':<10} {'TOTAL':<10}")
    lines.append("="*100)
    
    for result in results:
        matchup = f"{result.away_team} @ {result.home_team}"
        win_pct = f"{result.home_win_prob:.1f}%"
        score = f"{result.home_score_mean:.1f} - {result.away_score_mean:.1f}"
        spread = f"{result.spread:+.1f}"
        total = f"{result.total:.1f}"
        
        lines.append(f"{matchup:<30} {win_pct:<15} {score:<20} {spread:<10} {total:<10}")
    
    lines.append("="*100 + "\n")
    
    return '\n'.join(lines)


def get_betting_recommendation(result: PredictionResult, 
                               market_spread: float = None,
                               market_total: float = None) -> str:
    """
    Generate betting recommendations based on market lines
    
    Args:
        result: PredictionResult object
        market_spread: Current market spread (optional)
        market_total: Current market total (optional)
        
    Returns:
        Betting recommendation string
    """
    recommendations = []
    
    if market_spread is not None:
        edge = result.spread - market_spread
        if abs(edge) >= 2.0:
            if edge > 0:
                recommendations.append(f"VALUE: {result.home_team} ({edge:+.1f} edge)")
            else:
                recommendations.append(f"VALUE: {result.away_team} ({-edge:+.1f} edge)")
    
    if market_total is not None:
        edge = result.total - market_total
        if abs(edge) >= 3.0:
            if edge > 0:
                recommendations.append(f"VALUE: OVER ({edge:+.1f} edge)")
            else:
                recommendations.append(f"VALUE: UNDER ({-edge:+.1f} edge)")
    
    if not recommendations:
        return "No significant value detected"
    
    return " | ".join(recommendations)
