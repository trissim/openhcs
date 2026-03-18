from typing import Optional
from pathlib import Path

"""
Figure generation for the paper.
Generates matplotlib figures from benchmark results.
"""

def figure_srank_distribution(results, output_path: str = "figures/fig1_srank_dist.pdf"):
    """
    Figure 1: Distribution of srank across PDBbind complexes.
    Shows histogram of srank values and fraction of tractable cases.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figure generation")
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    sranks = [e.dq_srank for e in results.entries]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(sranks, bins=range(0, max(sranks) + 2), edgecolor='black', alpha=0.7)
    ax.set_xlabel("Structural Rank (srank)")
    ax.set_ylabel("Number of Complexes")
    ax.set_title("Distribution of srank across PDBbind Refined Set")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def figure_speedup_vs_srank(results, output_path: str = "figures/fig2_speedup.pdf"):
    """
    Figure 2: Speedup vs srank.
    Shows how DQ speedup over Vina depends on srank.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    sranks = [e.dq_srank for e in results.entries]
    speedups = [e.vina_time_s / max(e.dq_time_s, 1e-6) for e in results.entries]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(sranks, speedups, alpha=0.5, s=10)
    ax.set_xlabel("srank")
    ax.set_ylabel("Speedup (Vina time / DQ time)")
    ax.set_title("DQ-Dock Speedup vs Structural Rank")
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def figure_accuracy_comparison(results, output_path: str = "figures/fig3_accuracy.pdf"):
    """
    Figure 3: Accuracy comparison — DQ vs Vina vs Experimental.
    Scatter plot of predicted vs experimental binding affinity.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    exp = [e.experimental_affinity for e in results.entries]
    dq = [e.dq_affinity for e in results.entries]
    vina = [e.vina_affinity for e in results.entries]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(exp, dq, alpha=0.3, s=10, label='DQ-Dock')
    ax1.plot([min(exp), max(exp)], [min(exp), max(exp)], 'r--', label='Perfect')
    ax1.set_xlabel("Experimental pK")
    ax1.set_ylabel("DQ-Dock pK")
    ax1.set_title(f"DQ-Dock (RMSE={results.rmse_dq():.2f})")
    ax1.legend()
    
    ax2.scatter(exp, vina, alpha=0.3, s=10, label='Vina', color='orange')
    ax2.plot([min(exp), max(exp)], [min(exp), max(exp)], 'r--', label='Perfect')
    ax2.set_xlabel("Experimental pK")
    ax2.set_ylabel("Vina pK")
    ax2.set_title(f"AutoDock Vina (RMSE={results.rmse_vina():.2f})")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def figure_tractability_breakdown(results, output_path: str = "figures/fig4_tractability.pdf"):
    """
    Figure 4: Tractability breakdown.
    Pie chart of tractability classes + speedup per class.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    by_tract = results.by_tractability()
    labels = list(by_tract.keys())
    sizes = [len(v.entries) for v in by_tract.values()]
    speedups = [v.mean_speedup for v in by_tract.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax1.set_title("Distribution by Tractability Class")
    
    ax2.bar(labels, speedups)
    ax2.set_ylabel("Mean Speedup")
    ax2.set_title("Speedup by Tractability Class")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_figures(results, output_dir: str = "figures"):
    """Generate all paper figures."""
    figure_srank_distribution(results, f"{output_dir}/fig1_srank_dist.pdf")
    figure_speedup_vs_srank(results, f"{output_dir}/fig2_speedup.pdf")
    figure_accuracy_comparison(results, f"{output_dir}/fig3_accuracy.pdf")
    figure_tractability_breakdown(results, f"{output_dir}/fig4_tractability.pdf")
    print(f"All figures saved to {output_dir}/")
