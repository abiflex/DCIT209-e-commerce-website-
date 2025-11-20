"""
Data Management Module
Handles team data loading, validation, and export functionality
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


class DataManager:
    """Manages team data and prediction exports"""
    
    @staticmethod
    def load_default_teams() -> pd.DataFrame:
        """Load default team data"""
        return pd.DataFrame({
            'team': ['Hawks', 'Clippers', 'Grizzlies', 'Bucks', 'Magic', 
                    '76ers', 'Kings', 'Spurs'],
            'off_rating': [116.5, 114.7, 105.9, 114.1, 116.8, 114.5, 108.6, 115.5],
            'def_rating': [109.2, 120.4, 116.6, 119.1, 109.8, 114.0, 122.1, 113.2],
            'pace': [102.05, 96.93, 110.60, 100.21, 98.87, 99.30, 103.80, 101.50],
            'injury_adj': [-12, -12, -13, -14, -6, -15, -6, -11],
            'rest_days': [0, 1, 1, 2, 2, 1, 2, 3],
            'home_bonus': [3.1, 3.2, 3.3, 4.2, 3.6, 3.9, 3.5, 3.4]
        })
    
    @staticmethod
    def load_from_csv(filepath: str) -> pd.DataFrame:
        """
        Load team data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with team data
        """
        try:
            df = pd.read_csv(filepath)
            DataManager._validate_team_data(df)
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")
    
    @staticmethod
    def save_to_csv(teams_data: pd.DataFrame, filepath: str) -> None:
        """Save team data to CSV"""
        teams_data.to_csv(filepath, index=False)
    
    @staticmethod
    def _validate_team_data(df: pd.DataFrame) -> None:
        """Validate team data structure"""
        required_cols = ['team', 'off_rating', 'def_rating', 'pace', 
                        'injury_adj', 'rest_days', 'home_bonus']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    @staticmethod
    def export_predictions_csv(predictions: list, filepath: str) -> None:
        """
        Export predictions to CSV file
        
        Args:
            predictions: List of PredictionResult objects
            filepath: Output CSV file path
        """
        data = []
        for pred in predictions:
            data.append({
                'home_team': pred.home_team,
                'away_team': pred.away_team,
                'home_win_prob': f"{pred.home_win_prob:.1f}%",
                'away_win_prob': f"{pred.away_win_prob:.1f}%",
                'home_score': f"{pred.home_score_mean:.1f}",
                'away_score': f"{pred.away_score_mean:.1f}",
                'spread': f"{pred.spread:+.1f}",
                'total': f"{pred.total:.1f}",
                'confidence_95_lower': f"{pred.confidence_interval_95[0]:+.1f}",
                'confidence_95_upper': f"{pred.confidence_interval_95[1]:+.1f}",
                'simulations': pred.simulations
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def export_predictions_json(predictions: list, filepath: str, pretty: bool = True) -> None:
        """
        Export predictions to JSON file
        
        Args:
            predictions: List of PredictionResult objects
            filepath: Output JSON file path
            pretty: Whether to format JSON with indentation
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'predictions': []
        }
        
        for pred in predictions:
            data['predictions'].append({
                'matchup': f"{pred.away_team} @ {pred.home_team}",
                'home_team': pred.home_team,
                'away_team': pred.away_team,
                'probabilities': {
                    'home_win': round(pred.home_win_prob, 1),
                    'away_win': round(pred.away_win_prob, 1)
                },
                'predicted_scores': {
                    'home': round(pred.home_score_mean, 1),
                    'away': round(pred.away_score_mean, 1)
                },
                'spread': round(pred.spread, 1),
                'total': round(pred.total, 1),
                'confidence_interval_95': {
                    'lower': round(pred.confidence_interval_95[0], 1),
                    'upper': round(pred.confidence_interval_95[1], 1)
                },
                'simulations': pred.simulations
            })
        
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)
    
    @staticmethod
    def create_prediction_summary(predictions: list) -> Dict:
        """Create summary statistics for multiple predictions"""
        if not predictions:
            return {}
        
        return {
            'total_games': len(predictions),
            'avg_total_points': sum(p.total for p in predictions) / len(predictions),
            'avg_spread': sum(abs(p.spread) for p in predictions) / len(predictions),
            'closest_game': min(predictions, key=lambda p: abs(p.spread)),
            'biggest_favorite': max(predictions, key=lambda p: abs(p.spread))
        }
