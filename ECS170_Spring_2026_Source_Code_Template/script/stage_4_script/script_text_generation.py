import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
repo_root = os.path.abspath(os.path.join(project_root, '..'))
sys.path.insert(0, project_root)
os.chdir(current_dir)

from local_code.stage_4_code.Method_RNN_Generator import TextGenerationExperiment


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 4 word-level text generation.')
    parser.add_argument('--model', choices=['rnn', 'lstm', 'gru'], default='gru')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--max-vocab-size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument(
        '--data',
        default=os.path.join(repo_root, 'stage_4_data', 'stage_4_data', 'text_generation', 'data'),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    output_dir = f'../../result/stage_4_result/generation/{args.model}/'
    experiment = TextGenerationExperiment(
        data_path=args.data,
        output_dir=output_dir,
        cell_type=args.model,
        sequence_length=args.sequence_length,
        max_vocab_size=args.max_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epoch=args.epochs,
        seed=args.seed,
    )
    print('************ Start Stage 4 Text Generation ************')
    experiment.train()
    experiment.save_outputs([
        'what did the',
        'why did the',
        'how many people',
        'i went to',
        'do you know',
    ])
    print('************ Finish ************')
