import os
import sys

import torch
from torchviz import make_dot

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_4_code.Method_RNN_Classifier import RecurrentSentimentNet
from local_code.stage_4_code.Method_RNN_Generator import WordLanguageModel


OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result', 'figures')


def render_torchviz_graph(output, model, output_name):
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.graph_attr.update(
        rankdir='LR',
        label=output_name.replace('_', ' ').title(),
        labelloc='t',
        fontsize='24',
    )
    dot.node_attr.update(fontsize='10')
    dot.edge_attr.update(fontsize='9')
    dot.render(
        os.path.join(OUTPUT_DIR, output_name),
        format='png',
        cleanup=True,
    )


def plot_classification_architecture():
    torch.manual_seed(2)
    model = RecurrentSentimentNet(
        vocab_size=20000,
        cell_type='gru',
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        dropout=0.35,
        bidirectional=True,
    )
    model.eval()
    # Torchviz expands recurrent computation over time. A short dummy sequence
    # keeps the graph readable while preserving the actual trained layer sizes.
    dummy_tokens = torch.randint(low=1, high=20000, size=(1, 8), dtype=torch.long)
    dummy_lengths = torch.full((1,), 8, dtype=torch.long)
    logits = model(dummy_tokens, dummy_lengths)
    render_torchviz_graph(logits, model, 'classification_bigru_torchviz')


def plot_generation_architecture():
    torch.manual_seed(2)
    model = WordLanguageModel(
        vocab_size=4631,
        cell_type='gru',
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
    )
    model.eval()
    dummy_tokens = torch.randint(low=1, high=4631, size=(1, 8), dtype=torch.long)
    logits, _ = model(dummy_tokens)
    render_torchviz_graph(logits, model, 'generation_gru_torchviz')


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_classification_architecture()
    plot_generation_architecture()
    print(os.path.join(OUTPUT_DIR, 'classification_bigru_torchviz.png'))
    print(os.path.join(OUTPUT_DIR, 'generation_gru_torchviz.png'))
