"""
Visualization Module
Creates charts and graphs for prediction analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pathlib import Path
from .predictor import PredictionResult, NBAPredictor


class PredictionVisualizer:
    """Creates visualizations for NBA game predictions"""
    
    def __init__(self, style: str = 'darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        """
        sns.set_style(style)
        self.colors = {
            'home': '#1f77b4',
            'away': '#ff7f0e',
            'neutral': '#2ca02c'
        }
    
    def plot_score_distribution(self, result: PredictionResult, 
                               save_path: Optional[str] = None,
                               show: bool = True) -> None:
        """
        Plot score distribution for both teams
        
        Args:
            result: PredictionResult object
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Generate sample distributions
        home_scores = np.random.normal(result.home_score_mean, 
                                      result.home_score_std, 10000)
        away_scores = np.random.normal(result.away_score_mean, 
                                      result.away_score_std, 10000)
        
        # Plot home team distribution
        ax1.hist(home_scores, bins=50, alpha=0.7, color=self.colors['home'], 
                edgecolor='black', density=True)
        ax1.axvline(result.home_score_mean, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {result.home_score_mean:.1f}')
        ax1.set_xlabel('Points', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(f'{result.home_team} Score Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot away team distribution
        ax2.hist(away_scores, bins=50, alpha=0.7, color=self.colors['away'], 
                edgecolor='black', density=True)
        ax2.axvline(result.away_score_mean, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {result.away_score_mean:.1f}')
        ax2.set_xlabel('Points', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title(f'{result.away_team} Score Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'{result.away_team} @ {result.home_team}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_win_probability(self, result: PredictionResult,
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
        """
        Plot win probability as a bar chart
        
        Args:
            result: PredictionResult object
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        teams = [result.home_team, result.away_team]
        probabilities = [result.home_win_prob, result.away_win_prob]
        colors = [self.colors['home'], self.colors['away']]
        
        bars = ax.barh(teams, probabilities, color=colors, edgecolor='black', linewidth=2)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Win Probability (%)', fontsize=12)
        ax.set_title(f'Win Probability: {result.away_team} @ {result.home_team}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 110)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_spread_distribution(self, result: PredictionResult,
                                save_path: Optional[str] = None,
                                show: bool = True) -> None:
        """
        Plot spread distribution with confidence intervals
        
        Args:
            result: PredictionResult object
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Generate spread distribution
        spreads = np.random.normal(result.spread, 
                                  np.sqrt(result.home_score_std**2 + result.away_score_std**2),
                                  10000)
        
        ax.hist(spreads, bins=60, alpha=0.7, color=self.colors['neutral'], 
               edgecolor='black', density=True)
        
        # Add mean line
        ax.axvline(result.spread, color='red', linestyle='--', linewidth=2,
                  label=f'Mean Spread: {result.spread:+.1f}')
        
        # Add confidence interval
        ci_lower, ci_upper = result.confidence_interval_95
        ax.axvline(ci_lower, color='orange', linestyle=':', linewidth=2,
                  label=f'95% CI: [{ci_lower:+.1f}, {ci_upper:+.1f}]')
        ax.axvline(ci_upper, color='orange', linestyle=':', linewidth=2)
        
        # Add zero line
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Point Spread', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Spread Distribution: {result.away_team} @ {result.home_team}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_multiple_games(self, results: List[PredictionResult],
                           save_path: Optional[str] = None,
                           show: bool = True) -> None:
        """
        Create comparison chart for multiple games
        
        Args:
            results: List of PredictionResult objects
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        matchups = [f"{r.away_team}\n@\n{r.home_team}" for r in results]
        
        # Win probabilities
        home_probs = [r.home_win_prob for r in results]
        x = np.arange(len(matchups))
        bars = ax1.bar(x, home_probs, color=self.colors['home'], edgecolor='black', linewidth=1.5)
        ax1.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel('Home Win Probability (%)', fontsize=11)
        ax1.set_title('Win Probabilities', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(matchups, fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for bar, prob in zip(bars, home_probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Spreads
        spreads = [r.spread for r in results]
        colors_spread = [self.colors['home'] if s > 0 else self.colors['away'] for s in spreads]
        bars = ax2.bar(x, spreads, color=colors_spread, edgecolor='black', linewidth=1.5)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Point Spread', fontsize=11)
        ax2.set_title('Predicted Spreads', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(matchups, fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # Totals
        totals = [r.total for r in results]
        bars = ax3.bar(x, totals, color=self.colors['neutral'], edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Total Points', fontsize=11)
        ax3.set_title('Predicted Totals', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(matchups, fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{total:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Confidence intervals
        ci_ranges = [(r.confidence_interval_95[1] - r.confidence_interval_95[0]) 
                     for r in results]
        bars = ax4.bar(x, ci_ranges, color='purple', alpha=0.6, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('95% CI Range (Points)', fontsize=11)
        ax4.set_title('Prediction Uncertainty', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(matchups, fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('NBA Game Predictions Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_full_report(self, result: PredictionResult, 
                          output_dir: str = 'reports') -> None:
        """
        Generate complete visual report for a game
        
        Args:
            result: PredictionResult object
            output_dir: Directory to save reports
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        matchup = f"{result.away_team}_at_{result.home_team}".replace(' ', '_')
        
        print(f"\n📊 Generating visual report for {result.away_team} @ {result.home_team}...")
        
        self.plot_score_distribution(result, 
                                    f"{output_dir}/{matchup}_scores.png", 
                                    show=False)
        self.plot_win_probability(result, 
                                 f"{output_dir}/{matchup}_win_prob.png", 
                                 show=False)
        self.plot_spread_distribution(result, 
                                     f"{output_dir}/{matchup}_spread.png", 
                                     show=False)
        
        print(f"✓ Report complete! Files saved to '{output_dir}/' directory\n")
