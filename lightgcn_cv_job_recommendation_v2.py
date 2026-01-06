"""
LIGHTGCN: CV → JOB RECOMMENDATION (VERSION 2 - IMPROVED)
=========================================================
Cải thiện:
1. Deduplicate jobs (loại job trùng tên)
2. Better ranking: GNN score + Skill match + Category bonus
3. Normalize skill match by CV (không phải by job)
4. Company diversity (recommend từ nhiều companies)
5. Category filtering (filter theo ngành nghề)
"""

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

# ============================================
# 1. LOAD DATA
# ============================================
print("=" * 80)
print("📂 LOADING JOB GRAPH DATA")
print("=" * 80)

data = torch.load("job_graph_large.pt", weights_only=False)
metadata = torch.load("job_graph_large_metadata.pt", weights_only=False)

num_jobs = data["job"].num_nodes
num_skills = data["skill"].num_nodes
num_companies = data["company"].num_nodes

print("✅ Loaded graph:")
print(f"   Jobs: {num_jobs}")
print(f"   Skills: {num_skills}")
print(f"   Companies: {num_companies}")

# Get edges
job_skill_edges = data["job", "requires", "skill"].edge_index
job_company_edges = data["job", "belongs_to", "company"].edge_index

print(f"   Job-Skill edges: {job_skill_edges.shape[1]}")
print(f"   Job-Company edges: {job_company_edges.shape[1]}")

# ============================================
# 2. PREPROCESS - DEDUPLICATE JOBS (IMPROVED)
# ============================================
print("\n" + "=" * 80)
print("🧹 DEDUPLICATING JOBS (IMPROVED NORMALIZATION)")
print("=" * 80)

import re


def normalize_job_title(title: str) -> str:
    """
    Normalize job title để detect duplicates tốt hơn

    Examples:
    - "Backend Developer ( Python )" → "backend developer python"
    - "Backend Developer (Python)" → "backend developer python"
    - "Senior Full-stack Developer" → "senior fullstack developer"
    - "AI_ NLP Engineer" → "ai nlp engineer"
    """
    # Lowercase
    title = title.lower().strip()

    # Remove special characters but keep letters, numbers, spaces
    title = re.sub(r"[^\w\s]", " ", title)

    # Remove underscores
    title = title.replace("_", " ")

    # Normalize common variations
    replacements = {
        "full stack": "fullstack",
        "full-stack": "fullstack",
        "front end": "frontend",
        "front-end": "frontend",
        "back end": "backend",
        "back-end": "backend",
        "dev ops": "devops",
        "dev-ops": "devops",
        "qa qc": "qa",
        "sr ": "senior ",
        "jr ": "junior ",
        "mid ": "middle ",
    }

    for old, new in replacements.items():
        title = title.replace(old, new)

    # Remove extra whitespace
    title = " ".join(title.split())

    return title


# Group jobs by normalized title (find duplicates)
job_by_title = defaultdict(list)
for job_idx, job_info in enumerate(metadata["jobs_data"]):
    title = normalize_job_title(job_info["job_title"])
    job_by_title[title].append(job_idx)

# Show some examples of normalized titles
print("\n📝 Sample normalized titles:")
sample_titles = list(job_by_title.keys())[:5]
for title in sample_titles:
    original = metadata["jobs_data"][job_by_title[title][0]]["job_title"]
    print(f"   '{original}' → '{title}'")

# Show duplicates found
duplicates_found = [
    (title, indices) for title, indices in job_by_title.items() if len(indices) > 1
]
print(f"\n🔍 Found {len(duplicates_found)} duplicate groups:")
for title, indices in duplicates_found[:5]:  # Show first 5
    print(f"   '{title}': {len(indices)} copies")

# Keep only one job per unique title (the one with most skills)
unique_jobs = []
duplicate_to_unique = {}  # Map duplicate job_idx to unique job_idx

for title, job_indices in job_by_title.items():
    if len(job_indices) == 1:
        unique_jobs.append(job_indices[0])
        duplicate_to_unique[job_indices[0]] = job_indices[0]
    else:
        # Choose the one with most skills
        best_idx = max(
            job_indices,
            key=lambda j: len(metadata["jobs_data"][j].get("skills_list", [])),
        )
        unique_jobs.append(best_idx)
        for j in job_indices:
            duplicate_to_unique[j] = best_idx

unique_jobs_set = set(unique_jobs)
print("\n📊 Deduplication Results:")
print(f"   Original jobs: {num_jobs}")
print(f"   Unique jobs: {len(unique_jobs)}")
print(f"   Removed duplicates: {num_jobs - len(unique_jobs)}")

# ============================================
# 3. BUILD UNIFIED GRAPH
# ============================================
print("\n" + "=" * 80)
print("🔄 BUILDING UNIFIED GRAPH")
print("=" * 80)

# Offsets
SKILL_OFFSET = num_jobs
COMPANY_OFFSET = num_jobs + num_skills

total_nodes = num_jobs + num_skills + num_companies
print(f"Total nodes: {total_nodes}")

# Build edge index
js_src = job_skill_edges[0]
js_dst = job_skill_edges[1] + SKILL_OFFSET

jc_src = job_company_edges[0]
jc_dst = job_company_edges[1] + COMPANY_OFFSET

edge_index = torch.stack(
    [
        torch.cat([js_src, js_dst, jc_src, jc_dst]),
        torch.cat([js_dst, js_src, jc_dst, jc_src]),
    ],
    dim=0,
)

print(f"Total edges (bidirectional): {edge_index.shape[1]}")

# Build lookup dictionaries
job_to_skills = defaultdict(set)
skill_to_jobs = defaultdict(set)
job_to_company = {}
company_to_jobs = defaultdict(set)
job_to_category = {}

for i in range(job_skill_edges.shape[1]):
    job_idx = job_skill_edges[0, i].item()
    skill_idx = job_skill_edges[1, i].item()
    job_to_skills[job_idx].add(skill_idx)
    skill_to_jobs[skill_idx].add(job_idx)

for i in range(job_company_edges.shape[1]):
    job_idx = job_company_edges[0, i].item()
    company_idx = job_company_edges[1, i].item()
    job_to_company[job_idx] = company_idx
    company_to_jobs[company_idx].add(job_idx)

# Extract categories for each job
for job_idx, job_info in enumerate(metadata["jobs_data"]):
    cats = job_info.get("categories_list", [])
    if isinstance(cats, str):
        cats = [cats]
    job_to_category[job_idx] = set(cats) if cats else {"Other"}

print(f"Jobs with skills: {len(job_to_skills)}")
print(f"Jobs with company: {len(job_to_company)}")

# ============================================
# 4. TRAIN/TEST SPLIT
# ============================================
print("\n" + "=" * 80)
print("📊 CREATING TRAIN/TEST SPLIT")
print("=" * 80)

all_js_edges = []
for i in range(job_skill_edges.shape[1]):
    job_idx = job_skill_edges[0, i].item()
    skill_idx = job_skill_edges[1, i].item()
    all_js_edges.append((job_idx, skill_idx))

random.seed(42)
random.shuffle(all_js_edges)

train_ratio = 0.8
split_idx = int(len(all_js_edges) * train_ratio)

train_edges = all_js_edges[:split_idx]
test_edges = all_js_edges[split_idx:]

print(f"Train edges: {len(train_edges)}")
print(f"Test edges: {len(test_edges)}")

# Build train graph
train_js_src = torch.tensor([e[0] for e in train_edges])
train_js_dst = torch.tensor([e[1] for e in train_edges]) + SKILL_OFFSET

train_edge_index = torch.stack(
    [
        torch.cat([train_js_src, train_js_dst, jc_src, jc_dst]),
        torch.cat([train_js_dst, train_js_src, jc_dst, jc_src]),
    ],
    dim=0,
)

print(f"Train graph edges: {train_edge_index.shape[1]}")

train_job_skills = defaultdict(set)
for job_idx, skill_idx in train_edges:
    train_job_skills[job_idx].add(skill_idx)

test_job_skills = defaultdict(set)
for job_idx, skill_idx in test_edges:
    test_job_skills[job_idx].add(skill_idx)

# ============================================
# 5. LIGHTGCN MODEL
# ============================================
print("\n" + "=" * 80)
print("🧠 BUILDING LIGHTGCN MODEL")
print("=" * 80)


class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCN_Improved(nn.Module):
    """
    LightGCN with improved recommendation logic
    """

    def __init__(self, num_jobs, num_skills, num_companies, embedding_dim, num_layers):
        super().__init__()

        self.num_jobs = num_jobs
        self.num_skills = num_skills
        self.num_companies = num_companies
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.job_embedding = nn.Embedding(num_jobs, embedding_dim)
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim)
        self.company_embedding = nn.Embedding(num_companies, embedding_dim)

        self.conv = LightGCNConv()

        nn.init.normal_(self.job_embedding.weight, std=0.1)
        nn.init.normal_(self.skill_embedding.weight, std=0.1)
        nn.init.normal_(self.company_embedding.weight, std=0.1)

    def get_all_embeddings(self, edge_index):
        x = torch.cat(
            [
                self.job_embedding.weight,
                self.skill_embedding.weight,
                self.company_embedding.weight,
            ],
            dim=0,
        )

        emb_list = [x]
        for _ in range(self.num_layers):
            x = self.conv(x, edge_index)
            emb_list.append(x)

        final_emb = torch.stack(emb_list, dim=0).mean(dim=0)

        job_emb = final_emb[: self.num_jobs]
        skill_emb = final_emb[self.num_jobs : self.num_jobs + self.num_skills]
        company_emb = final_emb[self.num_jobs + self.num_skills :]

        return job_emb, skill_emb, company_emb

    def forward(self, edge_index, jobs, pos_skills, neg_skills):
        job_emb, skill_emb, _ = self.get_all_embeddings(edge_index)

        job_e = job_emb[jobs]
        pos_e = skill_emb[pos_skills]
        neg_e = skill_emb[neg_skills]

        pos_scores = (job_e * pos_e).sum(dim=1)
        neg_scores = (job_e * neg_e).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        reg_loss = (
            job_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)
        ) / (2 * len(jobs))

        return loss, reg_loss


# ============================================
# 6. IMPROVED RECOMMENDATION FUNCTION
# ============================================


def compute_combined_score(
    gnn_score: float,
    cv_skill_indices: set,
    job_skill_indices: set,
    cv_categories: set,
    job_categories: set,
    weights: dict = None,
) -> tuple:
    """
    Compute combined ranking score

    Args:
        gnn_score: Score từ GNN embeddings
        cv_skill_indices: Skills trong CV
        job_skill_indices: Skills job yêu cầu
        cv_categories: Categories CV mong muốn (có thể rỗng)
        job_categories: Categories của job
        weights: Weights cho từng component

    Returns:
        (combined_score, skill_match_rate, category_match)
    """
    if weights is None:
        weights = {
            "gnn": 0.5,  # GNN embedding similarity
            "skill": 0.35,  # Skill match rate
            "category": 0.15,  # Category bonus
        }

    # 1. Normalize GNN score (assume range 0-10)
    gnn_normalized = min(gnn_score / 10.0, 1.0)

    # 2. Skill match rate (normalized by CV skills, not job skills!)
    if len(cv_skill_indices) > 0:
        matched = cv_skill_indices & job_skill_indices
        skill_match_rate = len(matched) / len(cv_skill_indices)
    else:
        skill_match_rate = 0

    # 3. Category match bonus
    if cv_categories and job_categories:
        category_match = 1.0 if cv_categories & job_categories else 0.0
    else:
        category_match = 0.5  # Neutral if no category preference

    # Combined score
    combined = (
        weights["gnn"] * gnn_normalized
        + weights["skill"] * skill_match_rate
        + weights["category"] * category_match
    )

    return combined, skill_match_rate, category_match


def recommend_jobs_improved(
    model,
    edge_index,
    cv_skill_names: list,
    metadata: dict,
    job_to_skills: dict,
    job_to_company: dict,
    job_to_category: dict,
    unique_jobs_set: set,
    top_k: int = 10,
    category_filter: list = None,
    diversity_penalty: float = 0.3,  # Penalty cho jobs cùng company
    device: str = "cpu",
) -> list:
    """
    IMPROVED Job Recommendation Function

    Features:
    - Deduplicate jobs
    - Combined scoring (GNN + Skill Match + Category)
    - Company diversity
    - Category filtering

    Args:
        model: Trained LightGCN model
        edge_index: Graph edges
        cv_skill_names: List of skill names from CV
        metadata: Metadata dict
        job_to_skills: Job → Skills mapping
        job_to_company: Job → Company mapping
        job_to_category: Job → Category mapping
        unique_jobs_set: Set of unique job indices (no duplicates)
        top_k: Number of recommendations
        category_filter: List of preferred categories (None = all)
        diversity_penalty: Penalty for same company (0-1)
        device: torch device

    Returns:
        List of recommendation dicts
    """
    print(f"\n📝 CV Skills: {cv_skill_names}")

    # 1. Parse CV skills
    skill_indices = []
    matched_names = []
    unmatched = []

    skill_to_idx = metadata["skill_to_idx"]

    for name in cv_skill_names:
        name_lower = name.lower().strip()
        if name_lower in skill_to_idx:
            skill_indices.append(skill_to_idx[name_lower])
            matched_names.append(name_lower)
        else:
            # Partial match
            found = False
            for skill, idx in skill_to_idx.items():
                if name_lower in skill or skill in name_lower:
                    skill_indices.append(idx)
                    matched_names.append(skill)
                    found = True
                    break
            if not found:
                unmatched.append(name)

    print(f"   ✅ Matched skills ({len(matched_names)}): {matched_names}")
    if unmatched:
        print(f"   ⚠️ Unmatched skills: {unmatched}")

    if not skill_indices:
        print("   ❌ No skills matched!")
        return []

    cv_skill_set = set(skill_indices)
    cv_categories = set(category_filter) if category_filter else set()

    # 2. Get GNN scores for all jobs
    model.eval()
    with torch.no_grad():
        job_emb, skill_emb, company_emb = model.get_all_embeddings(
            edge_index.to(device)
        )

        # Aggregate CV skill embeddings
        cv_skill_embs = skill_emb[skill_indices]
        cv_vector = cv_skill_embs.mean(dim=0)

        # Compute similarity with all jobs
        gnn_scores = torch.matmul(job_emb, cv_vector).cpu().numpy()

    # 3. Compute combined scores for unique jobs only
    job_scores = []

    for job_idx in unique_jobs_set:
        job_skill_set = job_to_skills.get(job_idx, set())
        job_cat_set = job_to_category.get(job_idx, {"Other"})

        # Category filter
        if category_filter and not (cv_categories & job_cat_set):
            continue

        gnn_score = gnn_scores[job_idx]
        combined, skill_match, cat_match = compute_combined_score(
            gnn_score, cv_skill_set, job_skill_set, cv_categories, job_cat_set
        )

        job_scores.append(
            {
                "job_idx": job_idx,
                "gnn_score": gnn_score,
                "combined_score": combined,
                "skill_match_rate": skill_match,
                "category_match": cat_match,
            }
        )

    # 4. Sort by combined score
    job_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    # 5. Apply diversity (penalize same company)
    final_recommendations = []
    seen_companies = set()

    for job_data in job_scores:
        if len(final_recommendations) >= top_k:
            break

        job_idx = job_data["job_idx"]
        company_idx = job_to_company.get(job_idx)

        # Apply diversity penalty
        if company_idx in seen_companies:
            job_data["combined_score"] *= 1 - diversity_penalty

        # Re-check if still good enough
        if len(final_recommendations) < top_k:
            final_recommendations.append(job_data)
            if company_idx is not None:
                seen_companies.add(company_idx)

    # 6. Format output
    print(f"\n🎯 Top {top_k} Job Recommendations:")
    print("-" * 80)

    results = []
    for rank, job_data in enumerate(final_recommendations, 1):
        job_idx = job_data["job_idx"]
        job_info = metadata["jobs_data"][job_idx]
        company_idx = job_to_company.get(job_idx)
        company_name = (
            metadata["idx_to_company"].get(company_idx, "Unknown")
            if company_idx
            else "Unknown"
        )

        # Skill details
        job_skills = job_to_skills.get(job_idx, set())
        matched_skill_indices = cv_skill_set & job_skills
        missing_skills = job_skills - cv_skill_set

        matched_skill_names = [
            metadata["idx_to_skill"][s] for s in matched_skill_indices
        ]
        missing_skill_names = [
            metadata["idx_to_skill"][s] for s in list(missing_skills)[:5]
        ]

        # CV-based match rate (what % of CV skills does this job need?)
        cv_match_rate = (
            len(matched_skill_indices) / len(cv_skill_set) * 100 if cv_skill_set else 0
        )

        print(f"\n{rank}. {job_info['job_title']}")
        print(f"   🏢 Company: {company_name}")
        print(f"   📂 Category: {job_info.get('category', 'N/A')}")
        print(
            f"   💰 Salary: {job_info.get('salary_min', 0) / 1e6:.0f}M - {job_info.get('salary_max', 0) / 1e6:.0f}M VNĐ"
        )
        print(
            f"   📊 Combined Score: {job_data['combined_score']:.3f} | CV Match: {cv_match_rate:.0f}%"
        )
        print(
            f"   ✅ Your matching skills ({len(matched_skill_names)}): {', '.join(matched_skill_names[:5])}{'...' if len(matched_skill_names) > 5 else ''}"
        )

        if missing_skill_names:
            print(f"   📚 Skills to learn: {', '.join(missing_skill_names)}")

        results.append(
            {
                "rank": rank,
                "job_idx": job_idx,
                "job_title": job_info["job_title"],
                "company": company_name,
                "category": job_info.get("category", "N/A"),
                "combined_score": job_data["combined_score"],
                "gnn_score": job_data["gnn_score"],
                "skill_match_rate": job_data["skill_match_rate"],
                "cv_match_rate": cv_match_rate,
                "matched_skills": matched_skill_names,
                "skills_to_learn": missing_skill_names,
            }
        )

    return results


# ============================================
# 7. TRAINING
# ============================================
print("\n" + "=" * 80)
print("🚀 TRAINING LIGHTGCN")
print("=" * 80)

EMBEDDING_DIM = 64
NUM_LAYERS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 100
REG_WEIGHT = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = LightGCN_Improved(
    num_jobs=num_jobs,
    num_skills=num_skills,
    num_companies=num_companies,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_edge_index_device = train_edge_index.to(device)

all_skills_set = set(range(num_skills))


def sample_negative(job_idx, num_neg=1):
    pos_skills = train_job_skills.get(job_idx, set())
    neg_candidates = list(all_skills_set - pos_skills)
    return random.sample(neg_candidates, min(num_neg, len(neg_candidates)))


train_losses = []
print(f"\nTraining for {NUM_EPOCHS} epochs...")
print("-" * 60)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0

    random.shuffle(train_edges)

    for i in range(0, len(train_edges), BATCH_SIZE):
        batch = train_edges[i : i + BATCH_SIZE]

        jobs, pos_skills, neg_skills = [], [], []

        for job_idx, skill_idx in batch:
            neg = sample_negative(job_idx, 1)
            if neg:
                jobs.append(job_idx)
                pos_skills.append(skill_idx)
                neg_skills.append(neg[0])

        if not jobs:
            continue

        jobs = torch.tensor(jobs, device=device)
        pos_skills = torch.tensor(pos_skills, device=device)
        neg_skills = torch.tensor(neg_skills, device=device)

        optimizer.zero_grad()
        bpr_loss, reg_loss = model(
            train_edge_index_device, jobs, pos_skills, neg_skills
        )
        loss = bpr_loss + REG_WEIGHT * reg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

print("-" * 60)
print("✅ Training complete!")


# ============================================
# 8. DEMO RECOMMENDATIONS
# ============================================
print("\n" + "=" * 80)
print("🎯 CV → JOB RECOMMENDATION DEMOS (IMPROVED)")
print("=" * 80)

# === DEMO 1: Backend Developer ===
print("\n" + "=" * 70)
print("📋 DEMO 1: Backend Developer CV")
print("=" * 70)

backend_cv = ["python", "django", "docker", "postgresql", "rest api", "git", "linux"]
recommend_jobs_improved(
    model,
    train_edge_index_device,
    backend_cv,
    metadata,
    job_to_skills,
    job_to_company,
    job_to_category,
    unique_jobs_set,
    top_k=5,
    category_filter=["IT"],
    device=device,
)

# === DEMO 2: Data Analyst ===
print("\n" + "=" * 70)
print("📋 DEMO 2: Data Analyst CV")
print("=" * 70)

data_cv = ["python", "sql", "excel", "power bi", "data analysis", "statistics"]
recommend_jobs_improved(
    model,
    train_edge_index_device,
    data_cv,
    metadata,
    job_to_skills,
    job_to_company,
    job_to_category,
    unique_jobs_set,
    top_k=5,
    category_filter=["Data", "IT"],
    device=device,
)

# === DEMO 3: Frontend Developer ===
print("\n" + "=" * 70)
print("📋 DEMO 3: Frontend Developer CV")
print("=" * 70)

frontend_cv = ["javascript", "react", "html", "css", "typescript", "git", "nodejs"]
recommend_jobs_improved(
    model,
    train_edge_index_device,
    frontend_cv,
    metadata,
    job_to_skills,
    job_to_company,
    job_to_category,
    unique_jobs_set,
    top_k=5,
    category_filter=["IT"],
    device=device,
)

# === DEMO 4: No category filter ===
print("\n" + "=" * 70)
print("📋 DEMO 4: Backend CV (No Category Filter - More Diverse)")
print("=" * 70)

recommend_jobs_improved(
    model,
    train_edge_index_device,
    backend_cv,
    metadata,
    job_to_skills,
    job_to_company,
    job_to_category,
    unique_jobs_set,
    top_k=5,
    category_filter=None,
    device=device,
)


# ============================================
# 9. EVALUATION
# ============================================
print("\n" + "=" * 80)
print("📊 EVALUATING MODEL")
print("=" * 80)


def evaluate_improved(
    model, edge_index, test_dict, train_dict, unique_jobs, k_list=[5, 10, 20]
):
    """Evaluate with Hit@K and MRR"""
    model.eval()

    results = {f"Hit@{k}": [] for k in k_list}
    results["MRR"] = []  # Mean Reciprocal Rank

    test_jobs = [
        j
        for j in test_dict.keys()
        if j in unique_jobs and len(train_dict.get(j, set())) >= 3
    ]

    with torch.no_grad():
        job_emb, skill_emb, _ = model.get_all_embeddings(edge_index)

        for job_idx in test_jobs[:200]:
            cv_skills = list(train_dict[job_idx])
            if not cv_skills:
                continue

            cv_skill_embs = skill_emb[cv_skills]
            cv_vector = cv_skill_embs.mean(dim=0)

            # Only consider unique jobs
            scores = torch.matmul(job_emb, cv_vector).cpu().numpy()

            # Mask non-unique jobs
            for j in range(len(scores)):
                if j not in unique_jobs:
                    scores[j] = -float("inf")

            ranked_jobs = np.argsort(-scores)

            # Find rank of target job
            rank = np.where(ranked_jobs == job_idx)[0]
            if len(rank) > 0:
                rank = rank[0] + 1  # 1-indexed
                results["MRR"].append(1.0 / rank)
            else:
                results["MRR"].append(0)

            for k in k_list:
                hit = 1 if job_idx in ranked_jobs[:k] else 0
                results[f"Hit@{k}"].append(hit)

    return {m: np.mean(v) for m, v in results.items()}


eval_results = evaluate_improved(
    model, train_edge_index_device, test_job_skills, train_job_skills, unique_jobs_set
)

print("\n📈 Improved Evaluation Results:")
print("-" * 40)
for metric, value in eval_results.items():
    print(f"   {metric}: {value:.4f}")


# ============================================
# 10. COMPARISON ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("📊 ANALYSIS: Unique Jobs vs All Jobs")
print("=" * 80)

# Count jobs per category
category_counts = defaultdict(int)
for job_idx in unique_jobs_set:
    cats = job_to_category.get(job_idx, {"Other"})
    for cat in cats:
        category_counts[cat] += 1

print("\n📂 Jobs per Category (Unique only):")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    print(f"   {cat}: {count}")

# Count jobs per company
company_counts = defaultdict(int)
for job_idx in unique_jobs_set:
    company_idx = job_to_company.get(job_idx)
    if company_idx is not None:
        company_name = metadata["idx_to_company"].get(company_idx, "Unknown")
        company_counts[company_name] += 1

print("\n🏢 Top 10 Companies (by unique jobs):")
for company, count in sorted(company_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"   {company[:40]}: {count}")


# ============================================
# 11. VISUALIZATION
# ============================================
print("\n" + "=" * 80)
print("📊 CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training Loss
ax1 = axes[0]
ax1.plot(train_losses, color="#3498db", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Plot 2: Evaluation Metrics
ax2 = axes[1]
hit_metrics = {k: v for k, v in eval_results.items() if k.startswith("Hit")}
bars = ax2.bar(
    hit_metrics.keys(), hit_metrics.values(), color=["#2ecc71", "#3498db", "#9b59b6"]
)
ax2.set_ylabel("Score", fontsize=12)
ax2.set_title("Hit Rate (Improved)", fontsize=14, fontweight="bold")
ax2.set_ylim(0, 1)
for bar, val in zip(bars, hit_metrics.values()):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center",
        fontsize=11,
    )

# Plot 3: Category Distribution
ax3 = axes[2]
cats = list(category_counts.keys())[:8]
counts = [category_counts[c] for c in cats]
ax3.barh(cats, counts, color=plt.cm.Set3(np.linspace(0, 1, len(cats))))
ax3.set_xlabel("Number of Jobs", fontsize=12)
ax3.set_title("Jobs per Category", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("cv_job_recommendation_v2_results.png", dpi=150, bbox_inches="tight")
print("✅ Saved: cv_job_recommendation_v2_results.png")


# ============================================
# 12. SAVE MODEL
# ============================================
print("\n" + "=" * 80)
print("💾 SAVING MODEL")
print("=" * 80)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_jobs": num_jobs,
        "num_skills": num_skills,
        "num_companies": num_companies,
        "embedding_dim": EMBEDDING_DIM,
        "num_layers": NUM_LAYERS,
        "unique_jobs": list(unique_jobs_set),
        "train_losses": train_losses,
        "eval_results": eval_results,
    },
    "lightgcn_cv_job_model_v2.pt",
)

print("✅ Saved: lightgcn_cv_job_model_v2.pt")


# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("🎊 IMPROVED CV → JOB RECOMMENDATION COMPLETE!")
print("=" * 80)

print(f"""
📊 Improvements in V2:
   ✅ Deduplicated jobs: {num_jobs} → {len(unique_jobs_set)} unique jobs
   ✅ Combined scoring: GNN (50%) + Skill Match (35%) + Category (15%)
   ✅ Skill match normalized by CV (not by job)
   ✅ Company diversity penalty: {0.3 * 100:.0f}% for same company
   ✅ Category filtering supported

📈 Results:
   - Hit@5: {eval_results["Hit@5"]:.4f}
   - Hit@10: {eval_results["Hit@10"]:.4f}
   - Hit@20: {eval_results["Hit@20"]:.4f}
   - MRR: {eval_results["MRR"]:.4f}

🎯 Usage:
   recommend_jobs_improved(
       model, edge_index, 
       cv_skills=["python", "docker", "sql"],
       category_filter=["IT", "Data"],  # Optional
       top_k=10
   )

📁 Output files:
   - lightgcn_cv_job_model_v2.pt
   - cv_job_recommendation_v2_results.png
""")

plt.show()
