"""
KIỂM TRA FILE EXCEL db_base.xlsx
================================
Phân tích chi tiết dữ liệu để tìm vấn đề
"""

import numpy as np
import pandas as pd

# ============================================
# 1. ĐỌC FILE
# ============================================
print("=" * 80)
print("📂 READING db_base.xlsx")
print("=" * 80)

df = pd.read_excel("db_base.xlsx")

print(f"✅ Loaded {len(df)} rows x {len(df.columns)} columns")
print(f"\n📋 Columns: {df.columns.tolist()}")

# ============================================
# 2. KIỂM TRA GIÁ TRỊ NULL
# ============================================
print("\n" + "=" * 80)
print("🔍 NULL VALUES CHECK")
print("=" * 80)

null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)

print(f"{'Column':<25} {'Null Count':>12} {'Percentage':>12}")
print("-" * 80)
for col in df.columns:
    if null_counts[col] > 0:
        print(f"{col:<25} {null_counts[col]:>12} {null_pct[col]:>11}%")

# ============================================
# 3. KIỂM TRA DỮ LIỆU CƠ BẢN
# ============================================
print("\n" + "=" * 80)
print("📊 BASIC STATISTICS")
print("=" * 80)

print("\nUnique values per column:")
print("-" * 80)
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col:<25} {unique_count:>12}")

# ============================================
# 4. KIỂM TRA CATEGORY
# ============================================
print("\n" + "=" * 80)
print("📂 CATEGORY ANALYSIS")
print("=" * 80)

print(f"Total unique categories: {df['category'].nunique()}")

# Kiểm tra multi-category
print("\n🔍 Checking for multi-category jobs...")
multi_cat_comma = df["category"].str.contains(",", na=False).sum()
multi_cat_slash = df["category"].str.contains("/", na=False).sum()
multi_cat_dash = df["category"].str.contains("-", na=False).sum()

print(f"   Jobs with ',' in category: {multi_cat_comma}")
print(f"   Jobs with '/' in category: {multi_cat_slash}")
print(f"   Jobs with '-' in category: {multi_cat_dash}")

total_multi = df["category"].str.contains(",|/|-", na=False, regex=True).sum()
print(
    f"   Total jobs with separators: {total_multi} ({total_multi / len(df) * 100:.1f}%)"
)

print("\n📊 Top 15 categories:")
print(df["category"].value_counts().head(15))

print("\n📝 Examples of multi-category:")
multi_examples = df[df["category"].str.contains(",|/", na=False, regex=True)].head(5)
for idx, row in multi_examples.iterrows():
    print(f"   {row['job_title'][:40]:<40} → {row['category']}")

# ============================================
# 5. KIỂM TRA SALARY
# ============================================
print("\n" + "=" * 80)
print("💰 SALARY ANALYSIS")
print("=" * 80)

print("\nsalary_min:")
print(df["salary_min"].describe())

print("\nsalary_max:")
print(df["salary_max"].describe())

# Jobs có salary = 0
zero_min = (df["salary_min"] == 0).sum()
zero_max = (df["salary_max"] == 0).sum()
both_zero = ((df["salary_min"] == 0) & (df["salary_max"] == 0)).sum()

print("\n⚠️ Salary issues:")
print(f"   salary_min = 0: {zero_min} jobs ({zero_min / len(df) * 100:.1f}%)")
print(f"   salary_max = 0: {zero_max} jobs ({zero_max / len(df) * 100:.1f}%)")
print(f"   Both = 0: {both_zero} jobs ({both_zero / len(df) * 100:.1f}%)")

# Salary range
print("\n📊 Salary range:")
print(f"   Min salary: {df['salary_min'].min():,.0f} VNĐ")
print(f"   Max salary: {df['salary_max'].max():,.0f} VNĐ")

# ============================================
# 6. KIỂM TRA SKILLS
# ============================================
print("\n" + "=" * 80)
print("💡 SKILLS ANALYSIS")
print("=" * 80)

# Parse skills
all_skills = []
skill_counts = []

for idx, row in df.iterrows():
    if pd.notna(row["skills"]):
        skills = [s.strip() for s in str(row["skills"]).split(",")]
        all_skills.extend(skills)
        skill_counts.append(len(skills))
    else:
        skill_counts.append(0)

print(f"Total unique skills: {len(set(all_skills))}")
print(f"Average skills per job: {np.mean(skill_counts):.2f}")
print(f"Max skills in one job: {max(skill_counts)}")
print(f"Jobs without skills: {skill_counts.count(0)}")

print("\n📊 Top 20 most common skills:")
from collections import Counter

skill_counter = Counter(all_skills)
for skill, count in skill_counter.most_common(20):
    print(f"   {skill:<40} {count:>5} ({count / len(df) * 100:.1f}%)")

# ============================================
# 7. KIỂM TRA JOB_LEVEL
# ============================================
print("\n" + "=" * 80)
print("📈 JOB_LEVEL ANALYSIS")
print("=" * 80)

print(df["job_level"].value_counts())

# ============================================
# 8. KIỂM TRA EXPERIENCE_YEARS
# ============================================
print("\n" + "=" * 80)
print("📅 EXPERIENCE_YEARS ANALYSIS")
print("=" * 80)

print(df["experience_years"].describe())
print("\n📊 Distribution:")
print(df["experience_years"].value_counts().sort_index())

# ============================================
# 9. KIỂM TRA LOCATION
# ============================================
print("\n" + "=" * 80)
print("📍 LOCATION ANALYSIS")
print("=" * 80)

print(f"Total unique locations: {df['location_city'].nunique()}")
print("\nTop 10 locations:")
print(df["location_city"].value_counts().head(10))

# ============================================
# 10. KIỂM TRA COMPANY
# ============================================
print("\n" + "=" * 80)
print("🏢 COMPANY ANALYSIS")
print("=" * 80)

print(f"Total unique companies: {df['company_name'].nunique()}")
print("\nTop 10 companies:")
print(df["company_name"].value_counts().head(10))

print("\n📊 Company size distribution:")
print(df["company_size"].value_counts())

# ============================================
# 11. KIỂM TRA JOB_TYPE
# ============================================
print("\n" + "=" * 80)
print("💼 JOB_TYPE ANALYSIS")
print("=" * 80)

print(df["job_type"].value_counts())

# ============================================
# 12. TÓM TẮT VẤN ĐỀ
# ============================================
print("\n" + "=" * 80)
print("⚠️ SUMMARY OF ISSUES")
print("=" * 80)

issues = []

# 1. Null values
for col in df.columns:
    if null_counts[col] > 0:
        issues.append(
            f"❌ Column '{col}': {null_counts[col]} null values ({null_pct[col]}%)"
        )

# 2. Multi-category
if total_multi > 0:
    issues.append(f"⚠️ {total_multi} jobs have multi-category (need parsing)")

# 3. Salary = 0
if both_zero > 0:
    issues.append(f"⚠️ {both_zero} jobs have salary = 0 (need handling)")

# 4. Too many unique skills
if len(set(all_skills)) > 100:
    issues.append(f"⚠️ {len(set(all_skills))} unique skills (consider normalization)")

# 5. Jobs without skills
no_skills = skill_counts.count(0)
if no_skills > 0:
    issues.append(f"⚠️ {no_skills} jobs have no skills")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ No major issues found!")

# ============================================
# 13. ĐỀ XUẤT
# ============================================
print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

print("""
1. Multi-category: Cần parse và dùng multi-hot encoding
2. Salary = 0: Thay bằng giá trị median hoặc giữ nguyên với flag
3. Skills: Cần normalize (lowercase, gộp từ đồng nghĩa)
4. Null values: Cần xử lý (fill hoặc drop)
5. Company size: Cần chuẩn hóa format

➡️ Bạn có muốn tôi tạo code xử lý tất cả vấn đề này không?
""")

print("=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
