#!/usr/bin/env python3
"""
NBA Game Predictor - Command Line Interface
Professional NBA game prediction tool with Monte Carlo simulation
"""

import argparse
import sys
from pathlib import Path

from nba_predictor import NBAPredictor, PredictionVisualizer
from nba_predictor.data_manager import DataManager
from nba_predictor.utils import (
    format_prediction, 
    format_multiple_predictions,
    create_comparison_table,
    export_predictions
)


def main():
    parser = argparse.ArgumentParser(
        description='NBA Game Predictor - Monte Carlo Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single game
  python predict_nba.py Magic Clippers
  
  # Predict multiple games
  python predict_nba.py --games "Magic,Clippers" "Spurs,Hawks" "Bucks,76ers"
  
  # Export predictions to CSV
  python predict_nba.py --games "Magic,Clippers" "Spurs,Hawks" --export csv
  
  # Generate visualizations
  python predict_nba.py Magic Clippers --visualize
  
  # Detailed output with confidence intervals
  python predict_nba.py Magic Clippers --detailed
  
  # Use custom team data
  python predict_nba.py Magic Clippers --data teams.csv
        """
    )
    
    parser.add_argument('home', nargs='?', help='Home team name')
    parser.add_argument('away', nargs='?', help='Away team name')
    parser.add_argument('--games', nargs='+', metavar='HOME,AWAY',
                       help='Multiple games as "Home,Away" pairs')
    parser.add_argument('--data', type=str, metavar='FILE',
                       help='Load team data from CSV file')
    parser.add_argument('--simulations', type=int, default=40000,
                       help='Number of simulations (default: 40000)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--export', choices=['csv', 'json'],
                       help='Export predictions to file')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization charts')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for visualizations (default: reports)')
    parser.add_argument('--list-teams', action='store_true',
                       help='List all available teams')
    parser.add_argument('--team-info', type=str, metavar='TEAM',
                       help='Show detailed info for a team')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load team data
    try:
        if args.data:
            teams_data = DataManager.load_from_csv(args.data)
            print(f"✓ Loaded team data from: {args.data}\n")
        else:
            teams_data = DataManager.load_default_teams()
    except Exception as e:
        print(f"❌ Error loading team data: {e}")
        sys.exit(1)
    
    # Initialize predictor
    predictor = NBAPredictor(teams_data)
    
    # List teams
    if args.list_teams:
        print("\n📋 Available Teams:")
        print("=" * 50)
        for team in sorted(teams_data['team'].tolist()):
            print(f"  • {team}")
        print("=" * 50 + "\n")
        return
    
    # Show team info
    if args.team_info:
        try:
            info = predictor.get_team_summary(args.team_info)
            print(f"\n📊 Team Summary: {args.team_info}")
            print("=" * 50)
            print(f"  Offensive Rating:  {info['offensive_rating']:.1f}")
            print(f"  Defensive Rating:  {info['defensive_rating']:.1f}")
            print(f"  Net Rating:        {info['net_rating']:+.1f}")
            print(f"  Pace:              {info['pace']:.2f}")
            print(f"  Injury Adjustment: {info['injury_adjustment']:+.0f}")
            print(f"  Rest Days:         {info['rest_days']}")
            print(f"  Home Bonus:        {info['home_bonus']:+.1f}")
            print("=" * 50 + "\n")
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
        return
    
    # Predict games
    results = []
    
    try:
        if args.games:
            # Multiple games
            matchups = []
            for game in args.games:
                parts = game.split(',')
                if len(parts) != 2:
                    print(f"❌ Invalid format: '{game}'. Use 'Home,Away'")
                    sys.exit(1)
                matchups.append((parts[0].strip(), parts[1].strip()))
            
            print(f"\n🏀 Predicting {len(matchups)} games...")
            print(f"   Simulations per game: {args.simulations:,}\n")
            
            for home, away in matchups:
                result = predictor.predict(home, away, args.simulations, args.seed)
                results.append(result)
            
            # Display results
            if args.detailed:
                print(format_multiple_predictions(results, detailed=True))
            else:
                print(create_comparison_table(results))
        
        elif args.home and args.away:
            # Single game
            print(f"\n🏀 Predicting game...")
            print(f"   Simulations: {args.simulations:,}\n")
            
            result = predictor.predict(args.home, args.away, args.simulations, args.seed)
            results.append(result)
            
            print(format_prediction(result, detailed=args.detailed))
        
        else:
            parser.print_help()
            return
    
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    
    # Export predictions
    if args.export and results:
        try:
            export_predictions(results, format=args.export)
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    # Generate visualizations
    if args.visualize and results:
        try:
            visualizer = PredictionVisualizer()
            
            if len(results) == 1:
                visualizer.create_full_report(results[0], args.output_dir)
            else:
                Path(args.output_dir).mkdir(exist_ok=True)
                visualizer.plot_multiple_games(results, 
                                              f"{args.output_dir}/comparison.png",
                                              show=False)
                print(f"✓ Comparison chart saved to: {args.output_dir}/comparison.png\n")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            print("   Note: Ensure matplotlib and seaborn are installed")


if __name__ == '__main__':
    main()
