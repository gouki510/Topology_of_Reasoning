import argparse
import numpy as np
import json
import os

def display_sample_texts_per_cluster(dataset: str, model_name_or_path: str, output_dir: str, num_types: int, sample_count: int = 10):
    """
    各クラスタに対してサンプルテキストを数件表示する関数です。

    パラメータ:
      dataset: データセット名（例: 'gsm8k'）
      model_name_or_path: モデル名またはパス
      output_dir: クラスタリング結果が保存されているディレクトリ
      num_types: k-meansで生成されたクラスタ数
      sample_count: 各クラスタから表示するテキスト数（デフォルトは10件）
    """
    # 出力ディレクトリのパスを構成
    base_out_dir = f"{output_dir}/{model_name_or_path}/{dataset}"
    
    # stepテキストが保存された JSON ファイルのパス
    text_json = f"{base_out_dir}/{dataset}_text.json"
    if not os.path.exists(text_json):
        print(f"テキストファイル {text_json} が見つかりません。事前の抽出処理を確認してください。")
        return

    with open(text_json, 'r') as wf:
        solution_steps = np.array(json.load(wf))
    
    # クラスタリング結果（np.saveで保存した clusters.npy）のパス
    cluster_dir = f"{base_out_dir}/k-means-k={num_types}"
    clusters_file = f"{cluster_dir}/clusters.npy"
    if not os.path.exists(clusters_file):
        print(f"クラスタファイル {clusters_file} が見つかりません。クラスタリング処理を確認してください。")
        return

    clusters = np.load(clusters_file)
    
    unique_clusters = np.unique(clusters)
    print(f"全{len(unique_clusters)}個のクラスタが見つかりました。")
    print("=" * 80)

    # 各クラスタごとに
    for cluster_id in unique_clusters:
        # 現在のクラスタに属するインデックスを抽出
        cluster_indices = np.where(clusters == cluster_id)[0]
        texts_in_cluster = solution_steps[cluster_indices]
        total = len(texts_in_cluster)
        print(f"\n■ クラスタ {cluster_id} （全{total}件中、サンプル{min(sample_count, total)}件を表示）：")
        print("-" * 60)
        # サンプル件数だけ表示（ここでは先頭の件を表示。必要に応じてランダム抽出も可能）
        for text in texts_in_cluster[:sample_count]:
            print(text)
            print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="各クラスタから約10件のサンプルテキストを表示します。")
    parser.add_argument('--dataset', type=str, default='gsm8k', help='データセット名')
    parser.add_argument('--model_name_or_path', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', help='モデル名またはパス')
    parser.add_argument('--output_dir', type=str, default='load_data/generated_extract_steps', help='クラスタリング結果の出力ディレクトリ')
    parser.add_argument('--num_types', type=int, default=50, help='k-meansのクラスタ数')
    parser.add_argument('--sample_count', type=int, default=10, help='各クラスタから表示するテキスト数')
    args = parser.parse_args()
    
    display_sample_texts_per_cluster(args.dataset, args.model_name_or_path, args.output_dir, args.num_types, args.sample_count)
