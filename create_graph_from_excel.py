"""
TẠO GRAPH TỪ EXCEL ĐÃ LÀM SẠCH
================================
Input: db_base_cleaned.xlsx
Output: job_graph_large.pt, job_graph_large_metadata.pt
"""

import pickle

import pandas as pd
import torch
from torch_geometric.data import HeteroData

# ============================================
# 1. ĐỌC DỮ LIỆU ĐÃ LÀM SẠCH
# ============================================
print("=" * 80)
print("📂 LOADING CLEANED DATA")
print("=" * 80)

df = pd.read_excel("db_base_cleaned.xlsx")
print(f"✅ Loaded {len(df)} jobs")

# Load metadata from cleaning
with open("cleaning_metadata.pkl", "rb") as f:
    cleaning_meta = pickle.load(f)

print(f"   Skills: {len(cleaning_meta['common_skills'])}")
print(f"   Categories: {len(cleaning_meta['categories'])}")

# ============================================
# 2. PARSE LISTS (from string)
# ============================================
print("\n" + "=" * 80)
print("🔄 PARSING LISTS")
print("=" * 80)


def parse_list_string(s):
    """Parse string representation of list"""
    if pd.isna(s) or s == "":
        return []
    s = str(s)
    if s.startswith("[") and s.endswith("]"):
        # Already a list representation
        import ast

        return ast.literal_eval(s)
    else:
        # Comma separated
        return [x.strip() for x in s.split(",") if x.strip()]


df["skills_list"] = df["skills"].apply(parse_list_string)
df["categories_list"] = df["category"].apply(parse_list_string)

print("✅ Parsed skills and categories")

# ============================================
# 3. TẠO MAPPINGS
# ============================================
print("\n" + "=" * 80)
print("🗂️ CREATING MAPPINGS")
print("=" * 80)

# Skills
all_skills = set()
for skills in df["skills_list"]:
    all_skills.update(skills)
all_skills = sorted(all_skills)
skill_to_idx = {skill: idx for idx, skill in enumerate(all_skills)}
idx_to_skill = {idx: skill for skill, idx in skill_to_idx.items()}

print(f"Skills: {len(skill_to_idx)}")

# Categories
all_categories = set()
for cats in df["categories_list"]:
    all_categories.update(cats)
all_categories = sorted(all_categories)
category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}

print(f"Categories: {len(category_to_idx)}")

# Companies
all_companies = sorted(df["company_name"].unique())
company_to_idx = {company: idx for idx, company in enumerate(all_companies)}
idx_to_company = {idx: company for company, idx in company_to_idx.items()}

print(f"Companies: {len(company_to_idx)}")

# Other mappings
level_map = {
    "Intern": 0,
    "Fresher": 1,
    "Junior": 2,
    "Mid": 3,
    "Senior": 4,
    "Manager": 5,
}
job_type_map = {
    "Full-time": 0,
    "Part-time": 1,
    "Remote": 2,
    "Hybrid": 3,
    "Internship": 4,
}
location_map = {"Hanoi": 0, "HCM": 1, "Danang": 2, "Other": 3, "Unknown": 4}
company_size_map = {
    "1-9": 0,
    "10-24": 1,
    "25-99": 2,
    "100-499": 3,
    "500-1000": 4,
    "1000+": 5,
    "Unknown": 6,
}

# ============================================
# 4. TẠO HETERODATA GRAPH
# ============================================
print("\n" + "=" * 80)
print("🔨 CREATING HETERODATA GRAPH")
print("=" * 80)

data = HeteroData()

num_jobs = len(df)
num_skills = len(all_skills)
num_companies = len(all_companies)

print(f"   Jobs: {num_jobs}")
print(f"   Skills: {num_skills}")
print(f"   Companies: {num_companies}")

# --- 4.1 JOB NODE FEATURES ---
print("\n📊 Creating job node features...")

job_features = []
for idx, row in df.iterrows():
    # Multi-hot encode categories
    cat_vector = [0] * len(all_categories)
    for cat in row["categories_list"]:
        if cat in category_to_idx:
            cat_vector[category_to_idx[cat]] = 1

    # Other features
    features = (
        cat_vector  # Multi-hot categories
        + [
            level_map.get(row["job_level"], 0),
            row["experience_years"],
            row["salary_min"] / 1000000,  # Normalize to millions
            row["salary_max"] / 1000000,
            row["has_salary"],
            job_type_map.get(row["job_type"], 0),
            location_map.get(row["location_city"], 0),
        ]
    )
    job_features.append(features)

data["job"].x = torch.tensor(job_features, dtype=torch.float)
data["job"].num_nodes = num_jobs

feature_dim = len(all_categories) + 7
print(f"   Feature dimension: {feature_dim}")
print(f"   Shape: {data['job'].x.shape}")

# --- 4.2 SKILL NODE FEATURES ---
print("\n💡 Creating skill node features...")
data["skill"].x = torch.eye(num_skills, dtype=torch.float)
data["skill"].num_nodes = num_skills
print(f"   Shape: {data['skill'].x.shape}")

# --- 4.3 COMPANY NODE FEATURES ---
print("\n🏢 Creating company node features...")

# Company features: size
company_sizes = {}
for idx, row in df.iterrows():
    company_sizes[row["company_name"]] = company_size_map.get(row["company_size"], 0)

company_features = []
for company in all_companies:
    company_features.append([company_sizes.get(company, 0)])

data["company"].x = torch.tensor(company_features, dtype=torch.float)
data["company"].num_nodes = num_companies
print(f"   Shape: {data['company'].x.shape}")

# --- 4.4 EDGES: Job → Skill ---
print("\n🔗 Creating job-skill edges...")

job_skill_src = []
job_skill_dst = []

for job_idx, row in df.iterrows():
    for skill in row["skills_list"]:
        if skill in skill_to_idx:
            job_skill_src.append(job_idx)
            job_skill_dst.append(skill_to_idx[skill])

data["job", "requires", "skill"].edge_index = torch.tensor(
    [job_skill_src, job_skill_dst], dtype=torch.long
)
data["skill", "required_by", "job"].edge_index = torch.tensor(
    [job_skill_dst, job_skill_src], dtype=torch.long
)

print(f"   Job → Skill: {len(job_skill_src)} edges")

# --- 4.5 EDGES: Job → Company ---
print("\n🔗 Creating job-company edges...")

job_company_src = []
job_company_dst = []

for job_idx, row in df.iterrows():
    company = row["company_name"]
    if company in company_to_idx:
        job_company_src.append(job_idx)
        job_company_dst.append(company_to_idx[company])

data["job", "belongs_to", "company"].edge_index = torch.tensor(
    [job_company_src, job_company_dst], dtype=torch.long
)
data["company", "has_job", "job"].edge_index = torch.tensor(
    [job_company_dst, job_company_src], dtype=torch.long
)

print(f"   Job → Company: {len(job_company_src)} edges")

# ============================================
# 5. GRAPH SUMMARY
# ============================================
print("\n" + "=" * 80)
print("📊 GRAPH STRUCTURE")
print("=" * 80)
print(data)

print("\n📈 Statistics:")
print(f"   Total nodes: {data.num_nodes}")
print(f"   Total edges: {data.num_edges}")
print(f"   Node types: {data.node_types}")
print(f"   Edge types: {data.edge_types}")

# ============================================
# 6. TẠO METADATA
# ============================================
print("\n" + "=" * 80)
print("🗂️ CREATING METADATA")
print("=" * 80)

# Convert DataFrame to list of dicts for easier access
jobs_data = df.to_dict("records")

metadata = {
    "jobs_data": jobs_data,
    "skill_to_idx": skill_to_idx,
    "idx_to_skill": idx_to_skill,
    "company_to_idx": company_to_idx,
    "idx_to_company": idx_to_company,
    "category_to_idx": category_to_idx,
    "all_skills": all_skills,
    "all_companies": all_companies,
    "all_categories": list(all_categories),
    "level_map": level_map,
    "job_type_map": job_type_map,
    "location_map": location_map,
    "company_size_map": company_size_map,
    "num_jobs": num_jobs,
    "num_skills": num_skills,
    "num_companies": num_companies,
    "feature_info": {
        "job_features": f"{len(all_categories)} categories + 7 other features",
        "categories": list(all_categories),
    },
}

print("✅ Metadata created")

# ============================================
# 7. LƯU GRAPH VÀ METADATA
# ============================================
print("\n" + "=" * 80)
print("💾 SAVING GRAPH AND METADATA")
print("=" * 80)

torch.save(data, "job_graph_large.pt")
print("✅ Saved: job_graph_large.pt")

torch.save(metadata, "job_graph_large_metadata.pt")
print("✅ Saved: job_graph_large_metadata.pt")

# ============================================
# 8. THỐNG KÊ CUỐI CÙNG
# ============================================
print("\n" + "=" * 80)
print("🎊 GRAPH CREATION COMPLETE!")
print("=" * 80)

print(f"""
📊 Final Statistics:
   - Jobs: {num_jobs}
   - Skills: {num_skills}
   - Companies: {num_companies}
   - Categories: {len(all_categories)}
   - Job-Skill edges: {len(job_skill_src)}
   - Job-Company edges: {len(job_company_src)}

📁 Output files:
   - job_graph_large.pt (Graph data)
   - job_graph_large_metadata.pt (Metadata)

➡️ Next: Use load_graph_large.py to test and visualize
""")
