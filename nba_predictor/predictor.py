"""
NBA Game Prediction Engine
Enhanced Monte Carlo simulation for NBA game predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Container for prediction results"""
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    home_score_mean: float
    away_score_mean: float
    home_score_std: float
    away_score_std: float
    spread: float
    total: float
    confidence_interval_95: Tuple[float, float]
    simulations: int


class NBAPredictor:
    """NBA Game Prediction Model using Monte Carlo simulation"""
    
    LEAGUE_AVG_DEF_RATING = 110.0
    MIN_SCORE_FLOOR = 92
    DEFAULT_SCORE_VARIANCE = 10
    INJURY_IMPACT_FACTOR = 0.7
    REST_BONUS = 2
    REST_PENALTY = -2
    REST_THRESHOLD = 2
    
    def __init__(self, teams_data: pd.DataFrame, score_variance: float = DEFAULT_SCORE_VARIANCE):
        """
        Initialize predictor with team data
        
        Args:
            teams_data: DataFrame with team statistics
            score_variance: Standard deviation for score simulation (default: 10)
        """
        self.teams = teams_data.copy()
        self.score_variance = score_variance
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate team data has required columns"""
        required_cols = ['team', 'off_rating', 'def_rating', 'pace', 
                        'injury_adj', 'rest_days', 'home_bonus']
        missing = set(required_cols) - set(self.teams.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _get_team_data(self, team_name: str) -> pd.Series:
        """Get team data with validation"""
        team_data = self.teams[self.teams['team'] == team_name]
        if team_data.empty:
            available = ', '.join(self.teams['team'].tolist())
            raise ValueError(f"Team '{team_name}' not found. Available: {available}")
        return team_data.iloc[0]
    
    def _calculate_base_score(self, off_rating: float, pace: float) -> float:
        """Calculate base expected score from offensive rating and pace"""
        return (off_rating / 100) * pace
    
    def _apply_defensive_adjustment(self, score: float, opp_def_rating: float, pace: float) -> float:
        """Adjust score based on opponent's defensive rating"""
        return score - (opp_def_rating - self.LEAGUE_AVG_DEF_RATING) * pace / 200
    
    def _apply_rest_adjustment(self, rest_days: int) -> float:
        """Calculate rest day bonus/penalty"""
        return self.REST_BONUS if rest_days >= self.REST_THRESHOLD else self.REST_PENALTY
    
    def _apply_injury_adjustment(self, injury_adj: float) -> float:
        """Calculate injury impact"""
        return injury_adj * self.INJURY_IMPACT_FACTOR
    
    def predict(self, home_team: str, away_team: str, 
                simulations: int = 40000, seed: Optional[int] = None) -> PredictionResult:
        """
        Predict game outcome using Monte Carlo simulation
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            simulations: Number of simulations to run (default: 40000)
            seed: Random seed for reproducibility (optional)
            
        Returns:
            PredictionResult object with comprehensive statistics
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get team data
        home = self._get_team_data(home_team)
        away = self._get_team_data(away_team)
        
        # Calculate average pace
        pace = (home['pace'] + away['pace']) / 2
        
        # Calculate base scores using offensive ratings
        home_pts = self._calculate_base_score(home['off_rating'], pace)
        away_pts = self._calculate_base_score(away['off_rating'], pace)
        
        # Apply defensive adjustments
        home_pts = self._apply_defensive_adjustment(home_pts, away['def_rating'], pace)
        away_pts = self._apply_defensive_adjustment(away_pts, home['def_rating'], pace)
        
        # Apply home court advantage
        home_pts += home['home_bonus']
        
        # Apply rest adjustments
        home_pts += self._apply_rest_adjustment(home['rest_days'])
        away_pts += self._apply_rest_adjustment(away['rest_days'])
        
        # Apply injury adjustments
        home_pts += self._apply_injury_adjustment(home['injury_adj'])
        away_pts += self._apply_injury_adjustment(away['injury_adj'])
        
        # Apply score floor
        home_pts = max(home_pts, self.MIN_SCORE_FLOOR)
        away_pts = max(away_pts, self.MIN_SCORE_FLOOR)
        
        # Run Monte Carlo simulation
        home_scores = np.random.normal(home_pts, self.score_variance, simulations)
        away_scores = np.random.normal(away_pts, self.score_variance, simulations)
        
        # Calculate statistics
        home_wins = np.sum(home_scores > away_scores)
        home_win_prob = home_wins / simulations
        
        spreads = home_scores - away_scores
        spread_mean = np.mean(spreads)
        
        # Calculate 95% confidence interval for spread
        ci_lower = np.percentile(spreads, 2.5)
        ci_upper = np.percentile(spreads, 97.5)
        
        return PredictionResult(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win_prob * 100,
            away_win_prob=(1 - home_win_prob) * 100,
            home_score_mean=np.mean(home_scores),
            away_score_mean=np.mean(away_scores),
            home_score_std=np.std(home_scores),
            away_score_std=np.std(away_scores),
            spread=spread_mean,
            total=np.mean(home_scores + away_scores),
            confidence_interval_95=(ci_lower, ci_upper),
            simulations=simulations
        )
    
    def predict_multiple(self, matchups: list[Tuple[str, str]], 
                        simulations: int = 40000) -> list[PredictionResult]:
        """
        Predict multiple games
        
        Args:
            matchups: List of (home_team, away_team) tuples
            simulations: Number of simulations per game
            
        Returns:
            List of PredictionResult objects
        """
        return [self.predict(home, away, simulations) for home, away in matchups]
    
    def get_team_summary(self, team_name: str) -> Dict:
        """Get summary statistics for a team"""
        team = self._get_team_data(team_name)
        return {
            'team': team_name,
            'offensive_rating': team['off_rating'],
            'defensive_rating': team['def_rating'],
            'pace': team['pace'],
            'injury_adjustment': team['injury_adj'],
            'rest_days': team['rest_days'],
            'home_bonus': team['home_bonus'],
            'net_rating': team['off_rating'] - team['def_rating']
        }
