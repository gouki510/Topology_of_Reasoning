from tqdm import tqdm
import json, pickle
import argparse
import numpy as np
import transformers
import torch
from sklearn.cluster import KMeans
from utils import analyze_graph_v2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D, proj3d
from model.utils import model_name_mapping
from datasets import load_dataset
# TSNE
from sklearn.manifold import TSNE

import pandas as pd
import os
cbar3d = None

class Arrow3D(mpatches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super(Arrow3D, self).__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super(Arrow3D, self).draw(renderer)

    def do_3d_projection(self, renderer=None):
        # implement necessary methods for 3D projection to allow mplot3d to correctly project this object
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

GSMK_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.
The last line of your response should be of the following format: 'The answer is: ANSWER.' (without quotes) where ANSWER is just the final number or expression that solves the problem.
{Question}
""".strip()

def extract_step_type(dataset_name:str, model_name_or_path:str, batch_size:int, tokenizer_name_or_path:str, \
                      model_max_length = 1024, selection_method='k-means', output_dir='extract_steps', \
                      cache_dir=None, num_types=50, df_path:str=None, target_layer_ratio=0.5):

    out_dir = f"{output_dir}/{model_name_or_path}/{dataset_name}/target_layer_ratio={target_layer_ratio}"
    visualize_dir = f"{output_dir}/{model_name_or_path}/{dataset_name}/target_layer_ratio={target_layer_ratio}/visualize"
    
    model_name_or_path = model_name_mapping(model_name_or_path)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, legacy=False)
    tokenizer.model_max_length = model_max_length

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0

    # df_path == csv
    if df_path.endswith('.csv'):
        df = pd.read_csv(df_path)
    # df_path == json
    else:
        ds = load_dataset(df_path)
        df = pd.DataFrame(ds["train"])
    
    ### output files
    embedding_file = f"{out_dir}/{dataset_name}_embedding.npy"
    text_file = f"{out_dir}/{dataset_name}_text.npy"
    example_id_file = f"{out_dir}/{dataset_name}_example_id.npy"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(visualize_dir, exist_ok=True)

    ### load model
    embedding_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True, cache_dir=cache_dir,
        torch_dtype=torch.float16, device_map="auto")
    embedding_model.eval()
    
    target_layer = int(embedding_model.config.num_hidden_layers * target_layer_ratio)
    
    ### calculate embedding
    step_embeddings = []
    solution_steps = []
    example_ids = []
    ex_id = 0
    for index, row in tqdm(df.iterrows()):
        # print(batch)
        examples = []
        questions = []
        step_text = []
        
        if 'Question' in row:
            x = row['Question']
        elif 'question' in row:
            x = row['question']
        else:
            raise ValueError(f"Invalid column name: {df_path}")
        
        if 'generated_text' in row:
            steps = str(row['generated_text']).strip().split('\n')
        elif 'text' in row:
            steps = str(row['text']).strip().split('\n')
        elif 'solution' in row:
            steps = str(row['solution']).strip().split('\n')
        else:
            raise ValueError(f"Invalid column name: {df_path}")
        
        steps = [step.strip() for step in steps if len(step.strip())>5]
        # print("----steps----")
        # print(steps)
        if len(steps) > 1:
            questions.append(x.strip().split('\n'))
            examples.append(GSMK_QUERY_TEMPLATE.format(Question=x) + '\n'.join(steps[:-1]))
            step_text.append(steps)
        else:
            continue
        inputs = tokenizer(examples, return_tensors="pt", padding="longest", max_length=model_max_length, truncation=True).to('cuda')
        with torch.no_grad():
            outputs = embedding_model(**inputs, output_hidden_states=True, return_dict=True)
        nan_index = torch.where(torch.isnan(outputs.hidden_states[target_layer][0]))[0]
        # Get the last layer's hidden states
        # hidden_states has shape [layer, batch_size, seq_len, hidden_size]
        target_hidden_states = outputs.hidden_states[target_layer]
        
        vocab = tokenizer.get_vocab()

        # Collect token IDs that end with newline when decoded
        split_ids = [
            idx
            for tok, idx in vocab.items()
            if tokenizer.decode([idx], clean_up_tokenization_spaces=False).endswith("\n")
        ]
        split_ids_tensor = torch.tensor(split_ids, device=inputs['input_ids'].device)

        # Check if each element in inputs is in split_ids
        mask = torch.isin(inputs['input_ids'], split_ids_tensor)

        # step_mask : [batch_size, seq_len], becomes like [0,0,0,0,1,1,1,1,1,1,2,2,2...]
        step_mask = torch.cumsum(mask, dim=-1)
        step_mask *= inputs["attention_mask"]
        
        # Process each batch
        for hidden, mask, q, steps in zip(target_hidden_states, step_mask, questions, step_text):
            # hidden : [seq_len, hidden_size]
            # mask : [seq_len]
            # q : [1], Question text
            # steps : [Steps], Step text
            example_rep = []
            num_steps = torch.max(mask) + 1
            # Consider the newline in the question
            start = min(len(q), num_steps-1)
            for j in range(start, num_steps):
                step_j_mask = (mask == j).int().float()
                step_j_rep = (hidden * step_j_mask.unsqueeze(-1)).sum(0)
                step_len = step_j_mask.sum()
                if step_len > 0:
                    rep = (step_j_rep/step_len).cpu().numpy()
                    if np.isnan(rep).sum() == 0:
                        example_rep.append(rep)
                        solution_steps.append(steps[j-start])
                else:
                    assert False, "current step is empty"
            if len(example_rep) > 0:
                example_rep = np.stack(example_rep, axis=0)
                step_embeddings.append(example_rep)
                example_ids += [ex_id for _ in range(len(example_rep))]
                ex_id += 1
            else:
                assert False, "no step embeddings"
        
    step_embeddings = np.concatenate(step_embeddings, axis=0)
    solution_steps = np.array(solution_steps)
    example_ids = np.array(example_ids)
    
    # [all_steps, hidden_size]
    print("step_embeddings.shape: ", step_embeddings.shape)
    # [all_steps]
    print("solution_steps.shape: ", solution_steps.shape)
    # [all_steps]
    print("example_ids.shape: ", example_ids.shape)

    assert step_embeddings.shape[0] == solution_steps.shape[0] == example_ids.shape[0]
    np.save(embedding_file, step_embeddings)
    np.save(text_file, solution_steps)
    np.save(example_id_file, example_ids)

    
    with open(f"{out_dir}/{dataset_name}_text.json", 'w') as wf:
        json.dump(solution_steps.tolist(), wf)
    
    out_dir = f"{out_dir}/{selection_method}-k={num_types}"
    cluster_model_file = f"{out_dir}/{dataset_name}_{selection_method}_{num_types}.pkl"
    os.makedirs(f"{out_dir}", exist_ok=True)
    step_embeddings = np.float32(step_embeddings)
    # train_embeddings = deepcopy(step_embeddings)
    print("k-means start")
    cluster_model = KMeans(n_clusters=num_types, n_init=10, random_state=0).fit(step_embeddings)
    print("k-means end")
    print("cluster_model: ", cluster_model)
    print("cluster_model.labels_: ", cluster_model.labels_)
    print("cluster_model.cluster_centers_: ", cluster_model.cluster_centers_)

    with open(cluster_model_file, 'wb') as f:
        pickle.dump(cluster_model, f)

    all_preds = cluster_model.labels_
    print(f"all_preds.shape: {all_preds.shape}")
    assert len(all_preds) == len(solution_steps)

    np.save(f"{out_dir}/clusters.npy", all_preds)

    step_ids = np.arange(len(solution_steps))

    for i in range(num_types):
        # print(f"cluster {i}: ", np.sum(cluster_model.labels_==i))
        with open(f"{out_dir}/{dataset_name}_{num_types}_{i}.txt", 'w') as f:
            f.write('\n'.join(list(solution_steps[cluster_model.labels_==i])))
    tsne_file = f"{out_dir}/tsne.npy"

    X = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=3).fit_transform(np.float32(step_embeddings))
    np.save(tsne_file, X)

    plt.scatter(X[:, 0], X[:, 1], c=all_preds, s=2, cmap='viridis')
    plt.title(f"Number of Clusters = {num_types}")
    plt.savefig(f"{out_dir}/kmeans.png")
    plt.close()
    
    # center
    centers = cluster_model.cluster_centers_  # shape: (num_types, hidden_size)
    num_perplexity = 30
    tsne_on_centers = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=num_perplexity)
    center_tsne = tsne_on_centers.fit_transform(centers.astype(np.float32))
    
    tsne_3d_on_centers = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=num_perplexity)
    center_tsne_3d = tsne_3d_on_centers.fit_transform(centers.astype(np.float32))
    
    loop_detection_results = {}
    
    # generate Chain-of-Thought for each sample with batch size 1 and visualize with TSNE
    # for i, batch in tqdm(enumerate(dataloader), desc="Processing batches"):
    for index, row in tqdm(df.iterrows()):
        # print(batch)
        step_embeddings = []
        examples = []
        questions = []
        step_text = []
        
        if 'Question' in row:
            x = row['Question']
        elif 'question' in row:
            x = row['question']
        else:
            raise ValueError(f"Invalid column name: {df_path}")
        
        if 'generated_text' in row:
            steps = str(row['generated_text']).strip().split('\n')
        elif 'text' in row:
            steps = str(row['text']).strip().split('\n')
        elif 'solution' in row:
            steps = str(row['solution']).strip().split('\n')
        else:
            raise ValueError(f"Invalid column name: {df_path}")
        
        steps = [step.strip() for step in steps if len(step.strip())>5]
        out_txt = os.path.join(visualize_dir, f"{dataset_name}_steps_{index}.txt")

        # get the number of lines (number of steps)
        num_lines = len(steps)

        # write to the file
        with open(out_txt, "w", encoding="utf-8") as f:
            # first, write the number of lines (number of steps) to the first line
            # f.write(f"Number of steps: {num_lines}\n")
            # f.write("\n")  # insert an empty line if you want

            # then, write each element of steps with a newline
            for i, s in enumerate(steps, start=1):
                f.write(f"{i}: {s}\n")
                # print(f"{i}: {s}")

        # for debugging, print the number of steps
        print(f"Saved steps (count={num_lines}) to {out_txt}")
        if len(steps) > 1:
            questions.append(x.strip().split('\n'))
            examples.append(GSMK_QUERY_TEMPLATE.format(Question=x) + '\n'.join(steps[:-1]))
            step_text.append(steps)
        else:
            continue
        inputs = tokenizer(examples, return_tensors="pt", padding="longest", max_length=model_max_length, truncation=True).to('cuda')
        with torch.no_grad():
            outputs = embedding_model(**inputs, output_hidden_states=True, return_dict=True)
        nan_index = torch.where(torch.isnan(outputs.hidden_states[target_layer][0]))[0]
        # Get the last layer's hidden states
        # hidden_states has shape [layer, batch_size, seq_len, hidden_size]
        target_hidden_states = outputs.hidden_states[target_layer]
        
        vocab = tokenizer.get_vocab()

        # Collect token IDs that end with newline when decoded
        split_ids = [
            idx
            for tok, idx in vocab.items()
            if tokenizer.decode([idx], clean_up_tokenization_spaces=False).endswith("\n")
        ]
        split_ids_tensor = torch.tensor(split_ids, device=inputs['input_ids'].device)

        # Check if each element in inputs is in split_ids
        mask = torch.isin(inputs['input_ids'], split_ids_tensor)

        # step_mask : [batch_size, seq_len], becomes like [0,0,0,0,1,1,1,1,1,1,2,2,2...]
        step_mask = torch.cumsum(mask, dim=-1)
        step_mask *= inputs["attention_mask"]
        
        # Process each batch
        for hidden, mask, q, steps in zip(target_hidden_states, step_mask, questions, step_text):
            # hidden : [seq_len, hidden_size]
            # mask : [seq_len]
            # q : [1], Question text
            # steps : [Steps], Step text
            example_rep = []
            num_steps = torch.max(mask) + 1
            # Consider the newline in the question
            start = min(len(q), num_steps-1)
            for j in range(start, num_steps):
                step_j_mask = (mask == j).int().float()
                step_j_rep = (hidden * step_j_mask.unsqueeze(-1)).sum(0)
                step_len = step_j_mask.sum()
                if step_len > 0:
                    rep = (step_j_rep/step_len).cpu().numpy()
                    if np.isnan(rep).sum() == 0:
                        example_rep.append(rep)
                else:
                    assert False, "current step is empty"
            if len(example_rep) > 0:
                example_rep = np.stack(example_rep, axis=0)
                step_embeddings.append(example_rep)
            else:
                assert False, "no step embeddings"
        step_embeddings = np.concatenate(step_embeddings, axis=0)
        # print(f"step_embeddings.shape: {step_embeddings.shape}")

        # 1) Quick stats to see if something is off:
        print("Embedding stats → min:", np.nanmin(step_embeddings),
            "max:", np.nanmax(step_embeddings))

        # 2) Sanity‐check for non‐finite values:
        if not np.all(np.isfinite(step_embeddings)):
            bad_steps = np.where(~np.isfinite(step_embeddings).all(axis=1))[0]
            print(f"⚠️ Non‐finite embeddings at steps: {bad_steps}")
            # You can inspect them in more detail if you want:
            for idx in bad_steps:
                print(f"  step {idx} =>", step_embeddings[idx])

        # 3) Replace NaN/Inf (and clip extremes if you like):
        step_embeddings = np.nan_to_num(
            step_embeddings,
            nan=0.0,
            posinf=1e6,    # or np.finfo(np.float64).max
            neginf=-1e6    # or np.finfo(np.float64).min
        )
        # Optional: clip everything to a reasonable range
        step_embeddings = np.clip(step_embeddings, -1e5, 1e5)

        # 4) Ensure dtype & contiguity (you already had this):
        step_embeddings = np.require(step_embeddings,
                                    dtype=np.float32,
                                    requirements=['C'])
        # print(f"step_embeddings.shape: {step_embeddings.shape}")
        # now it should be safe to call:
        prompt_clusters = cluster_model.predict(step_embeddings)
        
        # distance of each step
        distance_list = []
        for i in range(len(step_embeddings)-1):
            distance_list.append(np.linalg.norm(step_embeddings[i] - step_embeddings[i+1]))
        distance_list = np.array(distance_list)
        print(f"Distance of each step: {distance_list}")
        assert len(distance_list) == len(prompt_clusters)-1

        print("\nKMeans Cluster Predictions for Prompt Steps:", prompt_clusters)
        
        ### Loop Detection
        loop_exists, loop_count, diameter, avg_clustering, avg_path_length, clustering_norm, path_length_norm, avg_hop_length, hop_length_norm, small_world_index = analyze_graph_v2(prompt_clusters, distance_list)
        print(f"Loop Detection: {'exists' if loop_exists else 'not exists'}")
        print(f"Number of complete loops: {loop_count}")
        print(f"Diameter: {diameter}")
        print(f"Average clustering coefficient: {avg_clustering}")
        print(f"Average path length: {avg_path_length}")
        print(f"Clustering normalization: {clustering_norm}")
        loop_detection_results[index] = {
            "loop_exists": loop_exists,
            "loop_count": loop_count,
            "diameter": diameter,
            "avg_clustering": avg_clustering,
            "avg_path_length": avg_path_length,
            "clustering_norm": clustering_norm,
            "path_length_norm": path_length_norm,
            "avg_hop_length": avg_hop_length,
            "hop_length_norm": hop_length_norm,
            "small_world_index": small_world_index
        }
        
        ### Visualize Reasoning Steps
        if len(step_embeddings) <= 1:
            continue
        # combine the background embedding and the step embedding and project with TSNE
        # combined_embeddings = step_embeddings
        # tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=min(30, max(5, len(step_embeddings) // 3)))
        # combined_tsne = tsne.fit_transform(combined_embeddings.astype(np.float32))
        # prompt_tsne = combined_tsne
        prompt_clusters = cluster_model.predict(step_embeddings)         # cluster ID for each step (length = number of steps)
        prompt_tsne     = center_tsne[prompt_clusters]   
        
        # ------------- 2D visualization (overall figure & zoomed figure) -------------
        n_steps = prompt_tsne.shape[0]

        # prepare the colormap and normalization (cover the range of step 0~n_steps-1)
        import matplotlib as mpl
        cmap2d = plt.get_cmap("viridis")
        norm2d = mpl.colors.Normalize(vmin=0, vmax=n_steps-1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # ------- left: overall figure -------
        ax_left = axes[0]
        # scatter plot: color corresponds to the step number
        scatter_left = ax_left.scatter(
            prompt_tsne[:, 0], prompt_tsne[:, 1],
            c=np.arange(n_steps),        # 0,1,2,...,n_steps-1
            cmap=cmap2d, norm=norm2d,
            s=120, edgecolors='black', zorder=3,
            label='Reasoning Steps'
        )

        # draw the arrow + text (cluster ID) for each step
        for j in range(n_steps - 1):
            x0, y0 = prompt_tsne[j]
            x1, y1 = prompt_tsne[j+1]
            arrow_color = cmap2d(norm2d(j))  # color corresponding to the starting step j
            ax_left.annotate(
                "",
                xy=(x1, y1), xycoords='data',
                xytext=(x0, y0), textcoords='data',
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2, shrinkA=5, shrinkB=5, alpha=0.8)
            )
            # display the cluster ID as a text label
            cluster_id = prompt_clusters[j]
            ax_left.text(x0, y0, str(cluster_id), fontsize=14, color='black', zorder=4)

        # put the cluster ID label on the final step
        x_last, y_last = prompt_tsne[-1]
        cluster_last = prompt_clusters[-1]
        ax_left.text(x_last, y_last, str(cluster_last), fontsize=14, color='black', zorder=4)

        ax_left.set_title("Chain-of-Thought: Full TSNE (Static)", fontsize=16)
        ax_left.set_xlabel("TSNE Dimension 1", fontsize=14)
        ax_left.set_ylabel("TSNE Dimension 2", fontsize=14)
        ax_left.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax_left.legend(loc='upper right', fontsize=12)
        cbar_left = fig.colorbar(scatter_left, ax=ax_left, pad=0.02)
        cbar_left.set_label("Reasoning Progress", fontsize=18, labelpad=10)
        cbar_left.ax.tick_params(labelsize=16)


        # ------- right: zoomed figure -------
        ax_right = axes[1]
        scatter_right = ax_right.scatter(
            prompt_tsne[:, 0], prompt_tsne[:, 1],
            c=np.arange(n_steps), cmap=cmap2d, norm=norm2d,
            s=120, edgecolors='black', zorder=3,
            label='Reasoning Steps'
        )

        for j in range(n_steps - 1):
            x0, y0 = prompt_tsne[j]
            x1, y1 = prompt_tsne[j+1]
            arrow_color = cmap2d(norm2d(j))
            ax_right.annotate(
                "",
                xy=(x1, y1), xycoords='data',
                xytext=(x0, y0), textcoords='data',
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2, shrinkA=5, shrinkB=5, alpha=0.8)
            )
            cluster_id = prompt_clusters[j]
            ax_right.text(x0, y0, str(cluster_id), fontsize=14, color='black', zorder=4)

        x_last, y_last = prompt_tsne[-1]
        cluster_last = prompt_clusters[-1]
        ax_right.text(x_last, y_last, str(cluster_last), fontsize=14, color='black', zorder=4)

        ax_right.set_title("Chain-of-Thought: Zoomed TSNE (Static)", fontsize=16)
        ax_right.set_xlabel("TSNE Dimension 1", fontsize=14)
        ax_right.set_ylabel("TSNE Dimension 2", fontsize=14)
        ax_right.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # add margin to the min/max of all steps
        x_min2, x_max2 = prompt_tsne[:, 0].min(), prompt_tsne[:, 0].max()
        y_min2, y_max2 = prompt_tsne[:, 1].min(), prompt_tsne[:, 1].max()
        margin_x2 = 0.1 * (x_max2 - x_min2)
        margin_y2 = 0.1 * (y_max2 - y_min2)
        ax_right.set_xlim(x_min2 - margin_x2, x_max2 + margin_x2)
        ax_right.set_ylim(y_min2 - margin_y2, y_max2 + margin_y2)

        cbar_right = fig.colorbar(scatter_right, ax=ax_right, pad=0.02)
        cbar_right.set_label("Step Index", fontsize=18)
        cbar_right.ax.tick_params(labelsize=16)

        plt.tight_layout()
        plt.savefig(f"{visualize_dir}/cot_trajectory_static_2d_{index}.pdf", dpi=300)
        plt.show()

        
        # ------------- 3D visualization -------------
        prompt_tsne_3d = center_tsne_3d[prompt_clusters]
        
        # prepare the colormap and normalization (use the same cmap2d, norm
        n_steps = prompt_tsne_3d.shape[0]
        # assert n_steps == num_lines, f"n_steps: {n_steps}, num_lines: {num_lines}"
        # if n_steps != num_lines:
        #     print("-"*100)
        #     print(f"n_steps: {n_steps}, num_lines: {num_lines}")
        #     print("-"*100)
        #     continue

        # set the style for the entire scene (dark background)
        # plt.style.use("dark_background")
        custom_style = {
            'figure.facecolor': '#000033',      # background color of the graph area (navy blue)
            'axes.facecolor': '#000033',        # background color of the graph (navy blue)
            'axes.edgecolor': '#00aaff',        # color of the graph frame (navy blue)
            'axes.labelcolor': '#00aaff',       # color of the axis label (navy blue)
            'text.color': 'white',            # color of the text (navy blue)
            'xtick.color': '#00aaff',           # color of the x-axis ticks (navy blue)
            'ytick.color': '#00aaff',           # color of the y-axis ticks (navy blue)
            'lines.color': '#00aaff',           # color of the line (navy blue)
            'patch.edgecolor': '#000033',       # color of the boundary line of the graph area (navy blue)
            'grid.color': '#003366',            # color of the grid line (dark navy blue)
            'grid.linestyle': '--',              # style
            'legend.edgecolor': '#00aaff',      # color of the legend frame (navy blue)
            'legend.facecolor': '#001f3f',      # background color of the legend (navy blue)
            'legend.framealpha': 0.5,           # transparency of the legend background
        }

        # apply the custom style    

        # prepare the colormap and normalization object
        cmap  = plt.get_cmap("viridis")
        norm  = mpl.colors.Normalize(vmin=0, vmax=n_steps-1)

        # calculate the axis range in advance and add margin
        x_min, x_max = prompt_tsne_3d[:, 0].min(), prompt_tsne_3d[:, 0].max()
        y_min, y_max = prompt_tsne_3d[:, 1].min(), prompt_tsne_3d[:, 1].max()
        z_min, z_max = prompt_tsne_3d[:, 2].min(), prompt_tsne_3d[:, 2].max()
        marg_x = 0.15 * (x_max - x_min) if (x_max != x_min) else 1.0
        marg_y = 0.15 * (y_max - y_min) if (y_max != y_min) else 1.0
        marg_z = 0.15 * (z_max - z_min) if (z_max != z_min) else 1.0
        x_lim = (x_min - marg_x, x_max + marg_x)
        y_lim = (y_min - marg_y, y_max + marg_y)
        z_lim = (z_min - marg_z, z_max + marg_z)

        # static 3D plot: more stylish
        fig3d = plt.figure(figsize=(12, 10), dpi=100)
        ax3d  = fig3d.add_subplot(111, projection='3d')

        # Set background color to slightly darker gray (correction from dark_background style)
        ax3d.xaxis.pane.fill      = True
        ax3d.xaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))

        ax3d.yaxis.pane.fill      = True
        ax3d.yaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))

        ax3d.zaxis.pane.fill      = True
        ax3d.zaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))

        # Remove grid lines if desired
        ax3d.grid(False)
        ax3d.grid(False)

        # Keep axis lines and ticks as simple as possible
        ax3d.xaxis.line.set_color((0.5, 0.5, 0.5, 1))
        ax3d.yaxis.line.set_color((0.5, 0.5, 0.5, 1))
        ax3d.zaxis.line.set_color((0.5, 0.5, 0.5, 1))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])

        # 3D scatter plot: color by step number, slightly larger marker size with a white border
        scatter3d = ax3d.scatter(
            prompt_tsne_3d[:, 0], prompt_tsne_3d[:, 1], prompt_tsne_3d[:, 2],
            c=np.arange(n_steps),            # Step number
            cmap=cmap, norm=norm,
            s=180,                           # Slightly larger
            edgecolors='w',                  # White border
            linewidths=0.8,
            depthshade=True
        )

        # Draw arrows + cluster ID labels
        for j in range(n_steps - 1):
            x0, y0, z0 = prompt_tsne_3d[j]
            x1, y1, z1 = prompt_tsne_3d[j+1]
            arrow_color = cmap(norm(j))
            arrow = Arrow3D(
                xs=[x0, x1], ys=[y0, y1], zs=[z0, z1],
                mutation_scale=25, lw=2.5, arrowstyle="-|>",
                color=arrow_color, alpha=0.9
            )
            ax3d.add_artist(arrow)
            # change the cluster ID label from red to more visible yellow
            cluster_id = prompt_clusters[j]
            ax3d.text(
                x0, y0, z0, str(cluster_id),
                fontsize=12, color='gold', fontweight='bold'
            )

        # Add cluster ID label to final step
        x_last, y_last, z_last = prompt_tsne_3d[-1]
        cluster_last = prompt_clusters[-1]
        ax3d.text(
            x_last, y_last, z_last, str(cluster_last),
            fontsize=12, color='gold', fontweight='bold'
        )

        # fix the axis ranges and make the title stylish
        ax3d.set_xlim(x_lim)
        ax3d.set_ylim(y_lim)
        ax3d.set_zlim(z_lim)
        if "Qwen2.5-32B" in model_name_or_path:
            ax3d.set_title("Reasoning Graph (Base Model)", fontsize=20, pad=30, color='lightgray')
        else:
            ax3d.set_title("Reasoning Graph (Reasoning Model)", fontsize=20, pad=30, color='lightgray')
        ax3d.set_xlabel("", fontsize=0)  # Hide labels
        ax3d.set_ylabel("", fontsize=0)
        ax3d.set_zlabel("", fontsize=0)

        # adjust the colorbar style
        cbar3d = fig3d.colorbar(scatter3d, ax=ax3d, pad=0.05, shrink=0.6)
        cbar3d.set_label("Reasoning Progress", fontsize=22, labelpad=30, color='lightgray')
        cbar3d.ax.yaxis.labelpad = 0.05
        cbar3d.ax.tick_params(labelsize=16, colors='lightgray')
        for spine in cbar3d.ax.spines.values():
            spine.set_edgecolor('lightgray')

        plt.tight_layout()
        plt.savefig(f"{visualize_dir}/cot_trajectory_static_3d_{index}_styled.pdf", dpi=300)
        plt.show()
        plt.close(fig3d)
        torch.cuda.empty_cache()
        
        # (Arrow3D class definition remains the same as the existing code)

        n_steps = prompt_tsne_3d.shape[0]

        # Prepare color normalization by "step number" in advance
        cmap  = plt.get_cmap("viridis")
        norm  = plt.Normalize(vmin=0, vmax=n_steps-1)

        # Create figure/axis for animation only once
        fig3d = plt.figure(figsize=(10, 8), dpi=80)
        ax3d  = fig3d.add_subplot(111, projection='3d')

        # Similarly set dark background & panel colors
        ax3d.xaxis.pane.fill      = True
        ax3d.xaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))

        ax3d.yaxis.pane.fill      = True
        ax3d.yaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))

        ax3d.zaxis.pane.fill      = True
        ax3d.zaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))

        # Remove grid lines if desired
        ax3d.grid(False)

        # Draw and cache initial point
        scatter3d = ax3d.scatter(
            [prompt_tsne_3d[0, 0]],
            [prompt_tsne_3d[0, 1]],
            [prompt_tsne_3d[0, 2]],
            c=[0], cmap=cmap, norm=norm,
            s=180, edgecolors='w', linewidths=0.8
        )

        # Colorbar is always displayed during animation (set initially)
        cbar3d = fig3d.colorbar(scatter3d, ax=ax3d, shrink=0.6)
        cbar3d.set_label("Reasoning Progress", fontsize=22, labelpad=30, color='lightgray')
        cbar3d.ax.yaxis.labelpad = 0.05
        cbar3d.ax.tick_params(labelsize=16, colors='lightgray')
        for spine in cbar3d.ax.spines.values():
            spine.set_edgecolor('lightgray')

        # Prepare lists for arrows and text
        arrows3d = []
        texts3d  = []

        # Fix axis ranges first
        ax3d.set_xlim(x_lim)
        ax3d.set_ylim(y_lim)
        ax3d.set_zlim(z_lim)

        # Set title and axis labels initially
        if "Qwen2.5" in model_name_or_path:
            ax3d.set_title("Reasoning Graph (Base Model)", fontsize=20, pad=20, color='lightgray')
        else:
            ax3d.set_title("Reasoning Graph (Reasoning Model)", fontsize=20, pad=20, color='lightgray')
        ax3d.set_xlabel("", fontsize=0)
        ax3d.set_ylabel("", fontsize=0)
        ax3d.set_zlabel("", fontsize=0)

        # Look at camera from slightly above and to the side
        ax3d.view_init(elev=30, azim=45)

        def update_3d(frame):
            # Add new point to scatter
            xs_old, ys_old, zs_old = scatter3d._offsets3d
            xs = np.append(xs_old, prompt_tsne_3d[frame, 0])
            ys = np.append(ys_old, prompt_tsne_3d[frame, 1])
            zs = np.append(zs_old, prompt_tsne_3d[frame, 2])
            scatter3d._offsets3d = (xs, ys, zs)

            # Also add step number to colors
            current_colors = list(scatter3d.get_array())
            current_colors.append(frame)
            scatter3d.set_array(np.array(current_colors))

            # Add arrow (when frame>0)
            if frame > 0:
                j = frame - 1
                x0, y0, z0 = prompt_tsne_3d[j]
                x1, y1, z1 = prompt_tsne_3d[j+1]
                arrow_color = cmap(norm(j))
                arrow = Arrow3D(
                    xs=[x0, x1], ys=[y0, y1], zs=[z0, z1],
                    mutation_scale=25, lw=3, arrowstyle="-|>",
                    color=arrow_color, alpha=0.9
                )
                ax3d.add_artist(arrow)
                arrows3d.append(arrow)

            # Rotate camera (view angle) slightly
            az = 45 + frame * 2
            ax3d.view_init(elev=30, azim=az)

            return (scatter3d, *arrows3d, *texts3d)

        # Create animation with FuncAnimation
        plt.style.use(custom_style)
        anim3d = FuncAnimation(
            fig3d,
            update_3d,
            frames=list(range(n_steps)),
            init_func=lambda: (),
            blit=True,    # Aim for differential update
            repeat=False
        )

        # Save as GIF
        gif_path = f"{visualize_dir}/cot_trajectory_3d_{index}_styled.gif"
        writer3d = PillowWriter(fps=8)  # Make it slightly smoother
        anim3d.save(gif_path, writer=writer3d)
        plt.close(fig3d)

        print(f"Saved 3D GIF → {gif_path}")
        
    # save results
    with open(f"{out_dir}results.json", 'w') as f:
        json.dump(loop_detection_results, f)


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gsm8k', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', help='model name or path')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', help='tokenizer name or path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--model_max_length', type=int, default=8192, help='model max length')
    parser.add_argument('--selection_method', type=str, default='k-means',  choices=['k-means'])
    parser.add_argument('--output_dir', type=str, default='results_20250606/generated_extract_steps', help='output dir')
    parser.add_argument('--cache_dir', type=str, default=None, help='cache dir')
    parser.add_argument('--num_types', type=int, default=200, help='number of reasoning types')
    parser.add_argument('--df_path', type=str, default=None, help='df path')
    parser.add_argument('--target_layer_ratio', type=float, default=0.5, help='target layer ratio')

    args = parser.parse_args()

    extract_step_type(args.dataset, args.model_name_or_path, args.batch_size, args.tokenizer_name_or_path,  
                      args.model_max_length, args.selection_method,
                      args.output_dir, args.cache_dir, args.num_types, args.df_path, args.target_layer_ratio)