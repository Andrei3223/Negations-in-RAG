import os
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/negation_controlled_with_answers')
    parser.add_argument('--output_dir', default=None, help='Defaults to same as data_dir')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    args = parser.parse_args()

    out_dir = args.output_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    for split_name in args.splits:
        in_path = os.path.join(args.data_dir, f'{split_name}.parquet')
        out_path = os.path.join(out_dir, f'{split_name}.csv')

        if not os.path.exists(in_path):
            print(f"  [{split_name}] not found: {in_path}, skipping")
            continue

        df = pd.read_parquet(in_path)
        df.to_csv(out_path, index=False)
        print(f"  [{split_name}] {len(df)} rows → {out_path}")
