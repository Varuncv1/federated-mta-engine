#!/usr/bin/env python3
"""
main.py: Orchestrator for the Federated Multi-Touch Attribution Engine pipeline.

Usage examples:
  python main.py simulate
  python main.py first-touch
  python main.py linear
  python main.py train-local --partner partner_a
  python main.py train-all-local
  python main.py federate
  python main.py flower-server
  python main.py flower-client --partner partner_a
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Federated MTA pipeline orchestrator"
    )
    subparsers = parser.add_subparsers(dest='command')

    # Simulate partner data
    subparsers.add_parser('simulate', help='Simulate partner impression & conversion logs')

    # Attribution heuristics: first-touch
    subparsers.add_parser('first-touch', help='Compute first-touch attribution credits')

    # Attribution heuristics: linear
    subparsers.add_parser('linear', help='Compute linear attribution credits')

    # Train a local model for one partner
    parser_tl = subparsers.add_parser('train-local', help='Train local model for a partner')
    parser_tl.add_argument('--partner', type=str, required=True,
                           help='Partner name (e.g., partner_a)')
    parser_tl.add_argument('--epochs', type=int, default=10)
    parser_tl.add_argument('--lr', type=float, default=0.01)
    parser_tl.add_argument('--batch_size', type=int, default=32)
    parser_tl.add_argument('--test_size', type=float, default=0.2)

    # Train all local models
    parser_tal = subparsers.add_parser('train-all-local', help='Train local models for all partners')
    parser_tal.add_argument('--epochs', type=int, default=10)
    parser_tal.add_argument('--lr', type=float, default=0.01)
    parser_tal.add_argument('--batch_size', type=int, default=32)
    parser_tal.add_argument('--test_size', type=float, default=0.2)

    # Federated aggregation (non-Flower)
    subparsers.add_parser('federate', help='Aggregate local models into a global model')

    # Flower federated server
    subparsers.add_parser('flower-server', help='Start Flower federated server')

    # Flower federated client for a partner
    parser_fc = subparsers.add_parser('flower-client', help='Start Flower client')
    parser_fc.add_argument('--partner', type=str, required=True,
                           help='Partner name for Flower client (e.g., partner_a)')

    args = parser.parse_args()

    # Ensure working directory is project root
    project_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(project_root)

    if args.command == 'simulate':
        from data.simulate_partners import main as simulate_main
        simulate_main()

    elif args.command == 'first-touch':
        import glob
        import pandas as pd
        from heuristics.first_touch import first_touch_attribution

        files = glob.glob(os.path.join('data', 'partner_*.csv'))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df_conv = df[df['converted'] == 1]
        credits = first_touch_attribution(df_conv)
        credits.to_csv('data/first_touch_credit.csv', index=False)
        print('First-touch credits saved to data/first_touch_credit.csv')

    elif args.command == 'linear':
        import glob
        import pandas as pd
        from heuristics.linear import linear_attribution

        files = glob.glob(os.path.join('data', 'partner_*.csv'))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df_conv = df[df['converted'] == 1]
        credits = linear_attribution(df_conv)
        credits.to_csv('data/linear_credit.csv', index=False)
        print('Linear credits saved to data/linear_credit.csv')

    elif args.command == 'train-local':
        from models.local_model import train_local_model
        train_local_model(
            partner=args.partner,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            test_size=args.test_size
        )

    elif args.command == 'train-all-local':
        from models.local_model import train_local_model
        partners = ['partner_a', 'partner_b', 'partner_c']
        for p in partners:
            print(f'Training local model for {p}...')
            train_local_model(
                partner=p,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                test_size=args.test_size
            )

    elif args.command == 'federate':
        from models.federated_trainer import main as federate_main
        federate_main()
    
    elif args.command == 'flower-server':
        # Launch the Flower server
        os.system("python fl/server.py")

    elif args.command == 'flower-client':
        # Launch a Flower client for the specified partner
        os.system(f"python fl/client.py {args.partner}")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
