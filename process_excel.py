"""
XỬ LÝ VÀ LÀM SẠCH DỮ LIỆU EXCEL
================================
Input: db_base.xlsx
Output: db_base_cleaned.xlsx
"""

import re
from collections import Counter

import pandas as pd

# ============================================
# 1. ĐỌC DỮ LIỆU
# ============================================
print("=" * 80)
print("📂 LOADING db_base.xlsx")
print("=" * 80)

df = pd.read_excel("db_base.xlsx")
print(f"✅ Loaded {len(df)} jobs")

# ============================================
# 2. XỬ LÝ SKILLS - NORMALIZE
# ============================================
print("\n" + "=" * 80)
print("💡 PROCESSING SKILLS")
print("=" * 80)


def normalize_skill(skill):
    """Chuẩn hóa 1 skill"""
    skill = skill.lower().strip()

    # Bỏ ký tự đặc biệt
    skill = re.sub(r"[^\w\s+#.-]", "", skill)

    # Mapping synonyms
    synonyms = {
        # Programming languages
        "javascript": ["js", "javascript", "java script"],
        "typescript": ["ts", "typescript", "type script"],
        "python": ["python", "python3", "py"],
        "java": ["java", "core java"],
        "csharp": ["c#", "csharp", "c sharp"],
        "cplusplus": ["c++", "cplusplus", "cpp"],
        # Frontend
        "react": ["react", "reactjs", "react.js", "react js"],
        "vue": ["vue", "vuejs", "vue.js", "vue js"],
        "angular": ["angular", "angularjs", "angular.js", "angular js"],
        "html": ["html", "html5"],
        "css": ["css", "css3"],
        # Backend
        "nodejs": ["node", "nodejs", "node.js", "node js"],
        "express": ["express", "expressjs", "express.js"],
        "django": ["django"],
        "flask": ["flask"],
        "spring": ["spring", "spring boot", "springboot"],
        # Database
        "sql": ["sql", "tsql", "t-sql"],
        "mysql": ["mysql", "my sql"],
        "postgresql": ["postgresql", "postgres", "psql"],
        "mongodb": ["mongodb", "mongo", "mongo db"],
        "redis": ["redis"],
        # DevOps
        "docker": ["docker"],
        "kubernetes": ["kubernetes", "k8s"],
        "jenkins": ["jenkins"],
        "git": ["git", "github", "gitlab"],
        "cicd": ["ci/cd", "cicd", "ci-cd"],
        "aws": ["aws", "amazon web services"],
        "azure": ["azure", "microsoft azure"],
        "gcp": ["gcp", "google cloud", "google cloud platform"],
        # Others
        "linux": ["linux", "ubuntu", "centos"],
        "restapi": ["rest api", "restful", "rest", "restapi"],
        "graphql": ["graphql", "graph ql"],
        "microservices": ["microservices", "micro services", "microservice"],
    }

    # Check synonyms
    for normalized, variants in synonyms.items():
        if skill in variants:
            return normalized

    # Special cases
    if "react" in skill and "native" not in skill:
        return "react"
    if "node" in skill:
        return "nodejs"
    if "angular" in skill:
        return "angular"
    if "vue" in skill:
        return "vue"

    return skill


def process_skills(skills_string):
    """Xử lý chuỗi skills"""
    if pd.isna(skills_string):
        return []

    # Split by comma
    skills = [s.strip() for s in str(skills_string).split(",")]

    # Normalize
    normalized = [normalize_skill(s) for s in skills if s.strip()]

    # Remove duplicates
    return list(set(normalized))


print("Normalizing skills...")
df["skills_list"] = df["skills"].apply(process_skills)

# Count all skills
all_skills = []
for skills in df["skills_list"]:
    all_skills.extend(skills)

skill_counter = Counter(all_skills)
print(f"Total unique skills before filter: {len(skill_counter)}")

# Filter: chỉ giữ skills xuất hiện >= 3 lần
min_frequency = 3
common_skills = {
    skill for skill, count in skill_counter.items() if count >= min_frequency
}

print(f"Skills appearing >= {min_frequency} times: {len(common_skills)}")

# Update skills_list
df["skills_list"] = df["skills_list"].apply(
    lambda skills: [s for s in skills if s in common_skills]
)

# Convert back to string
df["skills_clean"] = df["skills_list"].apply(lambda x: ", ".join(sorted(x)))

print(f"✅ Skills normalized: {len(skill_counter)} → {len(common_skills)}")

# ============================================
# 3. XỬ LÝ CATEGORY - MULTI-CATEGORY
# ============================================
print("\n" + "=" * 80)
print("📂 PROCESSING CATEGORY")
print("=" * 80)


def parse_category(category_string):
    """Parse multi-category"""
    if pd.isna(category_string):
        return ["Other"]

    category_string = str(category_string)

    # Split by comma, slash, or dash
    parts = re.split(r"[,/\-]", category_string)

    categories = []
    for part in parts:
        part = part.strip().lower()

        # Mapping to standard categories
        if any(
            keyword in part
            for keyword in ["it", "phần mềm", "software", "developer", "engineer"]
        ):
            categories.append("IT")
        elif any(keyword in part for keyword in ["kinh doanh", "sales", "bán hàng"]):
            categories.append("Sales")
        elif "marketing" in part:
            categories.append("Marketing")
        elif any(keyword in part for keyword in ["thiết kế", "design", "designer"]):
            categories.append("Design")
        elif any(keyword in part for keyword in ["data", "analyst", "phân tích"]):
            categories.append("Data")
        elif any(keyword in part for keyword in ["tester", "test", "qa", "qc"]):
            categories.append("QA")
        elif any(keyword in part for keyword in ["product", "manager", "quản lý"]):
            categories.append("Management")

    return list(set(categories)) if categories else ["Other"]


print("Parsing multi-category...")
df["categories_list"] = df["category"].apply(parse_category)
df["category_clean"] = df["categories_list"].apply(lambda x: ", ".join(sorted(x)))

category_counter = Counter([cat for cats in df["categories_list"] for cat in cats])
print("Categories distribution:")
for cat, count in category_counter.most_common():
    print(f"   {cat:<15} {count:>4} ({count / len(df) * 100:.1f}%)")

# ============================================
# 4. XỬ LÝ SALARY
# ============================================
print("\n" + "=" * 80)
print("💰 PROCESSING SALARY")
print("=" * 80)

# Fill null values with 0
df["salary_min"] = df["salary_min"].fillna(0)
df["salary_max"] = df["salary_max"].fillna(0)

# Create salary features
df["has_salary"] = ((df["salary_min"] > 0) | (df["salary_max"] > 0)).astype(int)

# GIỮ NGUYÊN salary = 0 (thỏa thuận)
# Chỉ normalize giá trị, không điền vào
df["salary_min_clean"] = df["salary_min"].fillna(0)
df["salary_max_clean"] = df["salary_max"].fillna(0)

print("\nSalary statistics:")
print(
    f"   Jobs with salary info: {df['has_salary'].sum()} ({df['has_salary'].sum() / len(df) * 100:.1f}%)"
)
print(
    f"   Jobs without salary (thỏa thuận): {(df['has_salary'] == 0).sum()} ({(df['has_salary'] == 0).sum() / len(df) * 100:.1f}%)"
)
print(
    f"   Salary range: {df[df['salary_max_clean'] > 0]['salary_min_clean'].min() / 1000000:.1f}M - {df['salary_max_clean'].max() / 1000000:.1f}M VNĐ"
)

# ============================================
# 5. XỬ LÝ COMPANY SIZE
# ============================================
print("\n" + "=" * 80)
print("🏢 PROCESSING COMPANY SIZE")
print("=" * 80)


def normalize_company_size(size_string):
    """Chuẩn hóa company size"""
    if pd.isna(size_string):
        return "Unknown"

    size_string = str(size_string).strip()

    # Map to standard ranges
    if "1-9" in size_string or size_string in ["1", "2", "3", "4", "5"]:
        return "1-9"
    elif "10-24" in size_string or size_string in ["10", "15", "20"]:
        return "10-24"
    elif "25-99" in size_string or size_string in ["25", "50", "75"]:
        return "25-99"
    elif "100-499" in size_string or size_string in ["100", "200", "300", "400"]:
        return "100-499"
    elif "500-1000" in size_string or size_string in ["500", "750", "1000"]:
        return "500-1000"
    elif size_string in ["5000", "10000"] or any(
        x in size_string for x in ["5000", "10000"]
    ):
        return "1000+"
    elif "1000" in size_string:
        return "1000+"
    else:
        return size_string


df["company_size_clean"] = df["company_size"].apply(normalize_company_size)

print("Company size distribution:")
print(df["company_size_clean"].value_counts())

# ============================================
# 6. XỬ LÝ JOB TYPE
# ============================================
print("\n" + "=" * 80)
print("💼 PROCESSING JOB TYPE")
print("=" * 80)


def normalize_job_type(job_type):
    """Chuẩn hóa job type"""
    if pd.isna(job_type):
        return "Full-time"

    job_type = str(job_type).lower()

    if "full" in job_type and "remote" in job_type:
        return "Remote"
    elif "full" in job_type and "hybrid" in job_type:
        return "Hybrid"
    elif "hybrid" in job_type:
        return "Hybrid"
    elif "remote" in job_type:
        return "Remote"
    elif "part" in job_type:
        return "Part-time"
    elif "intern" in job_type:
        return "Internship"
    else:
        return "Full-time"


df["job_type_clean"] = df["job_type"].apply(normalize_job_type)

print("Job type distribution:")
print(df["job_type_clean"].value_counts())

# ============================================
# 7. XỬ LÝ LOCATION
# ============================================
print("\n" + "=" * 80)
print("📍 PROCESSING LOCATION")
print("=" * 80)


def normalize_location(location):
    """Chuẩn hóa location"""
    if pd.isna(location):
        return "Unknown"

    location = str(location).lower()

    if "hà nội" in location or "hanoi" in location:
        return "Hanoi"
    elif "hồ chí minh" in location or "hcm" in location or "tp.hcm" in location:
        return "HCM"
    elif "đà nẵng" in location or "danang" in location:
        return "Danang"
    else:
        return "Other"


df["location_clean"] = df["location_city"].apply(normalize_location)

print("Location distribution:")
print(df["location_clean"].value_counts())

# ============================================
# 8. TẠO FINAL DATAFRAME
# ============================================
print("\n" + "=" * 80)
print("📦 CREATING CLEANED DATAFRAME")
print("=" * 80)

df_clean = pd.DataFrame(
    {
        "job_id": df["job_id"],
        "job_title": df["job_title"],
        "category": df["category_clean"],
        "categories_list": df["categories_list"],  # For processing
        "job_level": df["job_level"],
        "experience_years": df["experience_years"],
        "salary_min": df["salary_min_clean"],
        "salary_max": df["salary_max_clean"],
        "has_salary": df["has_salary"],
        "job_type": df["job_type_clean"],
        "skills": df["skills_clean"],
        "skills_list": df["skills_list"],  # For processing
        "location_city": df["location_clean"],
        "company_name": df["company_name"],
        "company_size": df["company_size_clean"],
    }
)

# ============================================
# 9. THỐNG KÊ CUỐI CÙNG
# ============================================
print("\n" + "=" * 80)
print("📊 FINAL STATISTICS")
print("=" * 80)

print(f"Total jobs: {len(df_clean)}")
print(f"Unique companies: {df_clean['company_name'].nunique()}")
print(f"Unique skills: {len(common_skills)}")
print(f"Unique categories: {len(category_counter)}")

print(f"\nAverage skills per job: {df_clean['skills_list'].apply(len).mean():.2f}")
print(f"Jobs with salary info: {df_clean['has_salary'].sum()}/{len(df_clean)}")

# ============================================
# 10. LƯU FILE
# ============================================
print("\n" + "=" * 80)
print("💾 SAVING CLEANED DATA")
print("=" * 80)

# Save to Excel
df_clean.to_excel("db_base_cleaned.xlsx", index=False)
print("✅ Saved: db_base_cleaned.xlsx")

# Save metadata
import pickle

metadata = {
    "common_skills": list(common_skills),
    "categories": list(category_counter.keys()),
    "total_jobs": len(df_clean),
}

with open("cleaning_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print("✅ Saved: cleaning_metadata.pkl")

print("\n" + "=" * 80)
print("🎉 DATA CLEANING COMPLETE!")
print("=" * 80)
print("\n➡️ Next: Run create_graph_from_excel.py to build graph")
