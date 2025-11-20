#!/usr/bin/env python3
"""
Example Usage - NBA Game Predictor
Demonstrates various ways to use the prediction system
"""

from nba_predictor import NBAPredictor, PredictionVisualizer
from nba_predictor.data_manager import DataManager
from nba_predictor.utils import format_prediction, create_comparison_table


def main():
    print("\n" + "="*70)
    print("  NBA GAME PREDICTOR - EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # Load default team data
    teams_data = DataManager.load_default_teams()
    
    # Initialize predictor
    predictor = NBAPredictor(teams_data)
    
    # Define tonight's games
    games = [
        ('Magic', 'Clippers'),
        ('Spurs', 'Hawks'),
        ('Bucks', '76ers'),
        ('Grizzlies', 'Kings')
    ]
    
    print(f"🏀 Predicting {len(games)} games with 40,000 simulations each...\n")
    
    # Predict all games
    results = []
    for home, away in games:
        result = predictor.predict(home, away, simulations=40000)
        results.append(result)
    
    # Display results in comparison table
    print(create_comparison_table(results))
    
    # Show detailed prediction for first game
    print("\n📊 DETAILED ANALYSIS - First Game:")
    print(format_prediction(results[0], detailed=True))
    
    # Export predictions
    print("\n💾 Exporting predictions...")
    DataManager.export_predictions_csv(results, 'predictions.csv')
    print("✓ Saved to: predictions.csv")
    
    DataManager.export_predictions_json(results, 'predictions.json')
    print("✓ Saved to: predictions.json")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    visualizer = PredictionVisualizer()
    
    # Create comparison chart for all games
    visualizer.plot_multiple_games(results, 'all_games_comparison.png', show=False)
    print("✓ Saved: all_games_comparison.png")
    
    # Create detailed report for first game
    visualizer.create_full_report(results[0], output_dir='reports')
    
    # Show team summary
    print("\n📋 TEAM SUMMARY - Magic:")
    print("="*70)
    summary = predictor.get_team_summary('Magic')
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:6.1f}")
        else:
            print(f"  {key:20s}: {value}")
    print("="*70)
    
    print("\n✅ Example complete! Check the generated files.\n")


if __name__ == '__main__':
    main()
