"""Build protein similarity graph from ESM-2 cosine similarity.

Steps:
  1. Load v2 dataset and extract ESM-2 embeddings.
  2. Compute pairwise cosine similarity.
  3. Run threshold sensitivity analysis.
  4. Build final graph at chosen threshold.
  5. Analyze graph structure (homophily, degree distribution, components).
  6. Save edge list and PyG data object for GNN training.

Usage:
  python build_graph.py                      # full pipeline with default threshold 0.99
  python build_graph.py --threshold 0.99     # explicit threshold
  python build_graph.py --sweep              # run threshold sensitivity sweep
"""

import argparse
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from data_utils import load_and_filter, get_esm_columns, build_pyg_data, create_masks


def compute_cosine_similarity(embeddings, batch_size=2000):
    """Compute pairwise cosine similarity in batches to manage memory.

    Args:
        embeddings: np.ndarray of shape [N, D].
        batch_size: number of rows per batch.

    Returns:
        Full N x N cosine similarity matrix (np.float32).
    """
    n = embeddings.shape[0]
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    num_batches = (n + batch_size - 1) // batch_size
    print(f"Computing cosine similarity: {n} proteins, {num_batches} batches...")

    for i in range(num_batches):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n)
        batch = embeddings[start_i:end_i]

        # Compute similarity of this batch against all rows
        sim_block = cosine_similarity(batch, embeddings)
        sim_matrix[start_i:end_i, :] = sim_block

        if (i + 1) % 5 == 0 or (i + 1) == num_batches:
            print(f"  Batch {i+1}/{num_batches} done")

    return sim_matrix


def threshold_to_edge_index(sim_matrix, threshold, max_edges=50_000_000):
    """Convert similarity matrix to PyG edge_index at given threshold.

    Only keeps edges where similarity > threshold (excludes self-loops).
    Raises ValueError if edge count exceeds max_edges to prevent OOM.

    Returns:
        edge_index: torch.LongTensor [2, num_edges]
        num_edges: int (unique undirected edges)
    """
    # Upper triangle to avoid duplicates
    rows, cols = np.where(np.triu(sim_matrix, k=1) > threshold)
    num_edges = len(rows)

    if num_edges > max_edges:
        raise ValueError(
            f"Threshold {threshold} produces {num_edges:,} edges (limit: {max_edges:,}). "
            f"Use a higher threshold."
        )

    # Undirected: add both directions
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index, num_edges


def count_edges_at_threshold(sim_matrix, threshold):
    """Count edges above threshold without materializing arrays (memory-safe)."""
    count = 0
    n = sim_matrix.shape[0]
    for i in range(n):
        # Only upper triangle: compare row i with columns i+1..n
        count += np.sum(sim_matrix[i, i+1:] > threshold)
    return int(count)


def analyze_graph(edge_index, num_nodes, y_labels, go_cols):
    """Analyze graph structure: components, degree distribution, homophily.

    Args:
        edge_index: torch.LongTensor [2, num_edges].
        num_nodes: int.
        y_labels: np.ndarray [num_nodes, num_go_terms] binary labels.
        go_cols: list of GO term names.
    """
    import networkx as nx

    print("\n--- Graph Structure Analysis ---")

    # Build NetworkX graph
    edges = edge_index.t().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges[:len(edges)//2])  # only unique edges (first half since we doubled)

    # Connected components
    components = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"Connected components: {len(components)}")
    print(f"Component sizes (top 10): {component_sizes[:10]}")
    isolated = sum(1 for s in component_sizes if s == 1)
    print(f"Isolated nodes: {isolated}")

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    degrees = np.array(degrees)
    print(f"\nDegree distribution:")
    print(f"  Mean: {degrees.mean():.1f}")
    print(f"  Median: {np.median(degrees):.1f}")
    print(f"  Max: {degrees.max()}")
    print(f"  Min: {degrees.min()}")
    print(f"  Std: {degrees.std():.1f}")
    print(f"  Nodes with degree 0: {(degrees == 0).sum()}")

    # Homophily ratio: fraction of edges where connected nodes share at least one GO term
    edge_list = edges[:len(edges)//2]  # unique edges
    if len(edge_list) > 0:
        shared_count = 0
        total_checked = len(edge_list)
        for u, v in edge_list:
            # Check if they share at least one GO term
            if np.any(y_labels[u] * y_labels[v]):
                shared_count += 1
        homophily = shared_count / total_checked
        print(f"\nHomophily (share >= 1 GO term): {homophily:.4f} ({shared_count}/{total_checked})")

        # Per-GO-term homophily (sample if too many edges)
        sample_size = min(100000, len(edge_list))
        if sample_size < len(edge_list):
            sample_idx = np.random.choice(len(edge_list), sample_size, replace=False)
            sampled_edges = edge_list[sample_idx]
        else:
            sampled_edges = edge_list

        # Average Jaccard similarity of GO labels across edges
        jaccard_sum = 0.0
        for u, v in sampled_edges:
            intersection = np.sum(y_labels[u] * y_labels[v])
            union = np.sum(np.clip(y_labels[u] + y_labels[v], 0, 1))
            if union > 0:
                jaccard_sum += intersection / union
        avg_jaccard = jaccard_sum / len(sampled_edges)
        print(f"Avg Jaccard similarity of GO labels across edges: {avg_jaccard:.4f}")
    else:
        print("\nNo edges — cannot compute homophily.")

    return {
        "components": len(components),
        "component_sizes_top5": component_sizes[:5],
        "isolated_nodes": isolated,
        "mean_degree": degrees.mean(),
        "median_degree": np.median(degrees),
        "max_degree": degrees.max(),
        "homophily": homophily if len(edge_list) > 0 else None,
        "avg_jaccard": avg_jaccard if len(edge_list) > 0 else None,
    }


def threshold_sweep(sim_matrix, thresholds, num_nodes, y_labels, go_cols, max_edges_for_analysis=20_000_000):
    """Run threshold sensitivity analysis.

    For each threshold, report: edges, components, mean degree, homophily.
    Uses memory-safe edge counting first; only builds full graph for thresholds
    with a manageable number of edges.
    """
    import networkx as nx

    print("\n=== Threshold Sensitivity Sweep ===")
    print(f"(Full graph analysis for thresholds with <= {max_edges_for_analysis:,} edges)")
    print(f"{'Threshold':>10} {'Edges':>12} {'Components':>12} {'Isolated':>10} {'MeanDeg':>10} {'Homophily':>10}")
    print("-" * 76)

    results = []
    for t in sorted(thresholds, reverse=True):  # start from highest (fewest edges)
        # Step 1: Count edges (memory-safe)
        num_edges = count_edges_at_threshold(sim_matrix, t)
        mean_deg = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        # Step 2: Full analysis only if edge count is manageable
        if num_edges <= max_edges_for_analysis and num_edges > 0:
            try:
                edge_index, _ = threshold_to_edge_index(sim_matrix, t, max_edges=max_edges_for_analysis)
                edges_np = edge_index.t().numpy()
                unique_edges = edges_np[:num_edges]

                G = nx.Graph()
                G.add_nodes_from(range(num_nodes))
                G.add_edges_from(unique_edges)

                components = nx.number_connected_components(G)
                degrees = np.array([d for _, d in G.degree()])
                isolated = int((degrees == 0).sum())

                # Homophily (sample for speed)
                sample_size = min(50000, len(unique_edges))
                sample_idx = np.random.choice(len(unique_edges), sample_size, replace=False)
                shared = sum(1 for u, v in unique_edges[sample_idx] if np.any(y_labels[u] * y_labels[v]))
                homophily = shared / sample_size

                del G, edges_np, edge_index  # free memory
            except (ValueError, MemoryError):
                components, isolated, homophily = "—", "—", "—"
        elif num_edges == 0:
            components, isolated, homophily = num_nodes, num_nodes, 0.0
        else:
            components, isolated, homophily = "—", "—", "—"

        # Format output
        comp_str = f"{components:>12,}" if isinstance(components, int) else f"{components:>12}"
        iso_str = f"{isolated:>10,}" if isinstance(isolated, int) else f"{isolated:>10}"
        hom_str = f"{homophily:>10.4f}" if isinstance(homophily, float) else f"{homophily:>10}"
        print(f"{t:>10.3f} {num_edges:>12,} {comp_str} {iso_str} {mean_deg:>10.1f} {hom_str}")

        results.append({
            "threshold": t,
            "edges": num_edges,
            "components": components if isinstance(components, int) else None,
            "isolated": isolated if isinstance(isolated, int) else None,
            "mean_degree": mean_deg,
            "homophily": homophily if isinstance(homophily, float) else None,
        })

    # Sort by threshold ascending for output
    results.sort(key=lambda r: r["threshold"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Build ESM-2 cosine similarity graph")
    parser.add_argument("--threshold", type=float, default=0.99, help="Cosine similarity threshold (default: 0.99)")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sensitivity sweep")
    parser.add_argument("--sweep-values", type=float, nargs="+", default=[0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99],
                        help="Threshold values for sweep")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for cosine similarity computation")
    parser.add_argument("--data-path", type=str, default="../data/final_df_with_features_expanded.csv")
    parser.add_argument("--output-dir", type=str, default="../data", help="Output directory for saved files")
    args = parser.parse_args()

    start_time = time.time()

    # Step 1: Load data
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)
    df, feature_cols, go_cols = load_and_filter(args.data_path)
    esm_cols = get_esm_columns(df)
    print(f"ESM embedding columns: {len(esm_cols)}")

    embeddings = df[esm_cols].values.astype(np.float32)
    y_labels = df[go_cols].values

    # Step 2: Compute cosine similarity
    print("\n" + "=" * 60)
    print("Step 2: Computing pairwise cosine similarity")
    print("=" * 60)
    sim_matrix = compute_cosine_similarity(embeddings, batch_size=args.batch_size)

    # Print similarity distribution
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(f"\nSimilarity distribution (upper triangle, {len(upper_tri):,} pairs):")
    for q in quantiles:
        print(f"  {q:.0%}: {np.quantile(upper_tri, q):.4f}")
    print(f"  Mean: {upper_tri.mean():.4f}, Std: {upper_tri.std():.4f}")
    print(f"  Min: {upper_tri.min():.4f}, Max: {upper_tri.max():.4f}")

    # Step 3: Threshold sweep (optional)
    if args.sweep:
        print("\n" + "=" * 60)
        print("Step 3: Threshold sensitivity sweep")
        print("=" * 60)
        sweep_results = threshold_sweep(sim_matrix, args.sweep_values, len(df), y_labels, go_cols)

        # Save sweep results
        sweep_df = pd.DataFrame(sweep_results)
        sweep_path = f"{args.output_dir}/threshold_sweep_results.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\nSweep results saved to {sweep_path}")

        # Check if chosen threshold is feasible
        edge_count = count_edges_at_threshold(sim_matrix, args.threshold)
        if edge_count > 50_000_000:
            print(f"\n[WARNING] Chosen threshold {args.threshold} produces {edge_count:,} edges (too many).")
            print(f"Skipping steps 4-6. Re-run with a higher --threshold (e.g., 0.99).")
            return

    # Step 4: Build graph at chosen threshold
    print("\n" + "=" * 60)
    print(f"Step 4: Building graph at threshold = {args.threshold}")
    print("=" * 60)
    edge_index, num_edges = threshold_to_edge_index(sim_matrix, args.threshold)
    print(f"Edges (undirected): {num_edges:,}")
    print(f"Edge index shape: {edge_index.shape}")

    # Step 5: Graph analysis
    print("\n" + "=" * 60)
    print("Step 5: Graph structure analysis")
    print("=" * 60)
    analyze_graph(edge_index, len(df), y_labels, go_cols)

    # Step 6: Build and save PyG data object
    print("\n" + "=" * 60)
    print("Step 6: Building PyG data object")
    print("=" * 60)
    data = build_pyg_data(df, feature_cols, go_cols, edge_index)
    data = create_masks(data)

    # Save
    save_path = f"{args.output_dir}/pyg_data_cosine_{args.threshold}.pt"
    torch.save(data, save_path)
    print(f"PyG data saved to {save_path}")

    # Also save edge list as CSV for reference
    edge_csv_path = f"{args.output_dir}/esm_cosine_edges_{args.threshold}.csv"
    unique_edges = edge_index.t().numpy()[:num_edges]
    edge_df = pd.DataFrame(unique_edges, columns=["src", "dst"])
    edge_df.to_csv(edge_csv_path, index=False)
    print(f"Edge list saved to {edge_csv_path}")

    # Save go_cols for later reference
    go_path = f"{args.output_dir}/go_terms.txt"
    with open(go_path, "w") as f:
        f.write("\n".join(go_cols))
    print(f"GO terms saved to {go_path}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
