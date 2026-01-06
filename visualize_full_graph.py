"""
VISUALIZE LARGE JOB GRAPH - FULL 3 NODE TYPES
==============================================
Visualize đầy đủ Job + Skill + Company
"""

from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

# ============================================
# 1. LOAD GRAPH
# ============================================

print("=" * 80)
print("📂 LOADING LARGE GRAPH")
print("=" * 80)

data = torch.load("job_graph_large.pt", weights_only=False)
metadata = torch.load("job_graph_large_metadata.pt", weights_only=False)

print("✅ Loaded!")
print(f"   Jobs: {data['job'].num_nodes}")
print(f"   Skills: {data['skill'].num_nodes}")
print(f"   Companies: {data['company'].num_nodes}")

# ============================================
# 2. STATISTICS
# ============================================

print("\n" + "=" * 80)
print("📊 GRAPH STATISTICS")
print("=" * 80)

# Top skills
edge_index = data["job", "requires", "skill"].edge_index
skill_connections = Counter(edge_index[1].tolist())
top_skills = skill_connections.most_common(15)

print("\n🔥 Top 15 Skills:")
for skill_idx, count in top_skills:
    skill_name = metadata["idx_to_skill"][skill_idx]
    print(f"   {skill_name:<25} {count:>3} jobs")

# Top companies
edge_index = data["job", "belongs_to", "company"].edge_index
company_connections = Counter(edge_index[1].tolist())
top_companies = company_connections.most_common(10)

print("\n🏢 Top 10 Companies:")
for company_idx, count in top_companies:
    company_name = metadata["idx_to_company"][company_idx]
    print(f"   {company_name[:35]:<35} {count:>3} jobs")

# ============================================
# 3. VISUALIZE FULL GRAPH (3 NODE TYPES)
# ============================================

print("\n" + "=" * 80)
print("🎨 CREATING FULL VISUALIZATION (Job + Skill + Company)")
print("=" * 80)

fig, ax = plt.subplots(figsize=(18, 14))
ax.set_title(
    "Job-Skill-Company Network (Full Graph Sample)",
    fontsize=18,
    fontweight="bold",
    pad=20,
)

G = nx.Graph()

# === Add top 15 skills ===
top_15_skills = [skill_idx for skill_idx, _ in top_skills[:15]]
for skill_idx in top_15_skills:
    skill_name = metadata["idx_to_skill"][skill_idx]
    G.add_node(f"S{skill_idx}", node_type="skill", label=skill_name, category="skill")

# === Add jobs connected to these skills (max 40) ===
job_indices = set()
job_skill_edges = data["job", "requires", "skill"].edge_index
for i in range(job_skill_edges.shape[1]):
    job_idx = job_skill_edges[0, i].item()
    skill_idx = job_skill_edges[1, i].item()

    if skill_idx in top_15_skills:
        job_indices.add(job_idx)
        if len(job_indices) >= 40:
            break

# Add job nodes
for job_idx in job_indices:
    job_info = metadata["jobs_data"][job_idx]
    G.add_node(
        f"J{job_idx}", node_type="job", label=job_info["job_title"][:20], category="job"
    )

# Add job-skill edges
for i in range(job_skill_edges.shape[1]):
    job_idx = job_skill_edges[0, i].item()
    skill_idx = job_skill_edges[1, i].item()

    if job_idx in job_indices and skill_idx in top_15_skills:
        G.add_edge(f"J{job_idx}", f"S{skill_idx}", etype="requires")

# === Add companies connected to these jobs ===
company_indices = set()
job_company_edges = data["job", "belongs_to", "company"].edge_index
for i in range(job_company_edges.shape[1]):
    job_idx = job_company_edges[0, i].item()
    company_idx = job_company_edges[1, i].item()

    if job_idx in job_indices:
        company_indices.add(company_idx)
        if f"C{company_idx}" not in G.nodes():
            company_name = metadata["idx_to_company"][company_idx]
            G.add_node(
                f"C{company_idx}",
                node_type="company",
                label=company_name[:18],
                category="company",
            )
        G.add_edge(f"J{job_idx}", f"C{company_idx}", etype="works_at")

print("\n📊 Graph Summary:")
print(f"   Total nodes: {G.number_of_nodes()}")
print(f"   - Jobs: {len(job_indices)}")
print(f"   - Skills: {len(top_15_skills)}")
print(f"   - Companies: {len(company_indices)}")
print(f"   Total edges: {G.number_of_edges()}")

# === Layout ===
pos = nx.spring_layout(G, k=2.0, iterations=80, seed=42)

# === Colors for 3 node types ===
node_colors = []
for node in G.nodes():
    ntype = G.nodes[node]["node_type"]
    if ntype == "job":
        node_colors.append("#FF6B6B")  # Red
    elif ntype == "skill":
        node_colors.append("#4ECDC4")  # Teal
    else:
        node_colors.append("#FFE66D")  # Yellow

# === Node sizes ===
node_sizes = []
for node in G.nodes():
    ntype = G.nodes[node]["node_type"]
    if ntype == "job":
        node_sizes.append(600)
    elif ntype == "skill":
        node_sizes.append(1200)
    else:
        node_sizes.append(1800)

# === Draw nodes ===
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.85,
    edgecolors="white",
    linewidths=1.5,
    ax=ax,
)

# === Draw edges với màu khác nhau ===
# Job-Skill edges (blue)
job_skill_edges_list = [
    (u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "requires"
]
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=job_skill_edges_list,
    alpha=0.4,
    edge_color="#3498db",
    width=1,
    ax=ax,
)

# Job-Company edges (orange)
job_company_edges_list = [
    (u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "works_at"
]
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=job_company_edges_list,
    alpha=0.5,
    edge_color="#e67e22",
    width=1.5,
    ax=ax,
)

# === Labels ===
labels = {node: G.nodes[node]["label"] for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

# === Legend ===
legend_elements = [
    mpatches.Patch(
        facecolor="#FF6B6B", edgecolor="white", label=f"Job ({len(job_indices)})"
    ),
    mpatches.Patch(
        facecolor="#4ECDC4", edgecolor="white", label=f"Skill ({len(top_15_skills)})"
    ),
    mpatches.Patch(
        facecolor="#FFE66D",
        edgecolor="white",
        label=f"Company ({len(company_indices)})",
    ),
    mpatches.Patch(
        facecolor="white", edgecolor="#3498db", label="requires (Job→Skill)"
    ),
    mpatches.Patch(
        facecolor="white", edgecolor="#e67e22", label="works_at (Job→Company)"
    ),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=11, framealpha=0.9)

ax.axis("off")
plt.tight_layout()
plt.savefig("full_graph_3nodes.png", dpi=200, bbox_inches="tight", facecolor="white")
print("\n✅ Saved: full_graph_3nodes.png")

# ============================================
# 4. CATEGORY DISTRIBUTION PIE CHART
# ============================================

fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

# === Pie 1: Categories ===
ax1 = axes[0]
ax1.set_title("Job Categories Distribution", fontsize=14, fontweight="bold")

categories = []
counts = []
for cat in metadata["all_categories"]:
    count = sum(1 for job in metadata["jobs_data"] if cat in job["categories_list"])
    categories.append(cat)
    counts.append(count)

colors_pie = plt.cm.Set3(np.linspace(0, 1, len(categories)))
wedges, texts, autotexts = ax1.pie(
    counts, labels=categories, colors=colors_pie, autopct="%1.1f%%", startangle=90
)
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight("bold")
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(10)
    autotext.set_fontweight("bold")

# === Pie 2: Job Levels ===
ax2 = axes[1]
ax2.set_title("Job Levels Distribution", fontsize=14, fontweight="bold")

job_levels = {}
for job in metadata["jobs_data"]:
    level = job.get("job_level", "Unknown")
    job_levels[level] = job_levels.get(level, 0) + 1

labels2 = list(job_levels.keys())
sizes2 = list(job_levels.values())
colors2 = plt.cm.Pastel1(np.linspace(0, 1, len(labels2)))
wedges2, texts2, autotexts2 = ax2.pie(
    sizes2, labels=labels2, colors=colors2, autopct="%1.1f%%", startangle=90
)
for text in texts2:
    text.set_fontsize(11)
    text.set_fontweight("bold")
for autotext in autotexts2:
    autotext.set_color("white")
    autotext.set_fontsize(10)
    autotext.set_fontweight("bold")

plt.tight_layout()
plt.savefig("category_distribution.png", dpi=150, bbox_inches="tight")
print("✅ Saved: category_distribution.png")

print("\n" + "=" * 80)
print("🎉 VISUALIZATION COMPLETE!")
print("=" * 80)
print("\n📁 Output files:")
print("   - full_graph_3nodes.png (Main graph với 3 loại nodes)")
print("   - category_distribution.png (Pie charts)")

plt.show()
