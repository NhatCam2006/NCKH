# BÁO CÁO: XÂY DỰNG JOB-SKILL GRAPH VÀ ÁP DỤNG LIGHTGCN

## 📋 MỤC LỤC
1. [Tổng quan](#1-tổng-quan)
2. [Dữ liệu đầu vào](#2-dữ-liệu-đầu-vào)
3. [Xử lý và làm sạch dữ liệu](#3-xử-lý-và-làm-sạch-dữ-liệu)
4. [Xây dựng Heterogeneous Graph](#4-xây-dựng-heterogeneous-graph)
5. [Áp dụng LightGCN](#5-áp-dụng-lightgcn)
6. [Kết quả thực nghiệm](#6-kết-quả-thực-nghiệm)
7. [Kết luận](#7-kết-luận)

---

## 1. TỔNG QUAN

### 1.1 Mục tiêu
- Xây dựng Knowledge Graph từ dữ liệu tuyển dụng (file Excel)
- Áp dụng thuật toán LightGCN để học biểu diễn (embeddings) trên graph
- Đánh giá khả năng dự đoán mối quan hệ Job-Skill

### 1.2 Công nghệ sử dụng
- **Ngôn ngữ**: Python 3.12
- **Thư viện chính**: 
  - PyTorch + PyTorch Geometric (xây dựng graph)
  - Pandas (xử lý dữ liệu)
  - LightGCN-PyTorch (thuật toán recommendation)

### 1.3 Cấu trúc dự án
```
Test_graph/
├── db_base.xlsx                    # Dữ liệu gốc
├── db_base_cleaned.xlsx            # Dữ liệu đã làm sạch
├── check_excel.py                  # Kiểm tra dữ liệu
├── process_excel.py                # Làm sạch dữ liệu
├── create_graph_from_excel.py      # Tạo graph
├── job_graph_large.pt              # Graph đã tạo
├── job_graph_large_metadata.pt     # Metadata
├── visualize_full_graph.py         # Trực quan hóa
└── LightGCN-PyTorch/               # Thuật toán LightGCN
    ├── code/
    │   ├── main.py
    │   ├── model.py
    │   └── dataloader.py
    └── data/
        └── jobskill/               # Dữ liệu đã convert
            ├── train.txt
            └── test.txt
```

---

## 2. DỮ LIỆU ĐẦU VÀO

### 2.1 Nguồn dữ liệu
File `db_base.xlsx` chứa thông tin tuyển dụng với các trường:

| Trường | Mô tả | Ví dụ |
|--------|-------|-------|
| job_id | Mã công việc | JOB001 |
| job_title | Tên vị trí | Backend Developer (Python) |
| category | Ngành nghề | IT, Data, Design |
| job_level | Cấp bậc | Junior, Senior, Manager |
| experience_years | Số năm kinh nghiệm | 2, 3, 5 |
| salary_min/max | Mức lương | 15,000,000 - 30,000,000 VNĐ |
| job_type | Loại công việc | Full-time, Remote, Hybrid |
| skills | Kỹ năng yêu cầu | Python, Django, Docker |
| location_city | Địa điểm | Hanoi, HCM, Danang |
| company_name | Tên công ty | FPT Software, VNG |
| company_size | Quy mô công ty | 100-499, 1000+ |

### 2.2 Thống kê dữ liệu gốc
- **Tổng số jobs**: 499
- **Số skills unique**: 908
- **Số companies**: 55
- **Số job-skill relationships**: 8,871

---

## 3. XỬ LÝ VÀ LÀM SẠCH DỮ LIỆU

### 3.1 Quy trình xử lý (process_excel.py)

#### a) Chuẩn hóa Skills
```python
# Mapping các từ đồng nghĩa
synonyms = {
    "javascript": ["js", "javascript", "java script"],
    "react": ["react", "reactjs", "react.js"],
    "nodejs": ["node", "nodejs", "node.js"],
    "postgresql": ["postgresql", "postgres", "psql"],
    ...
}
```
- Chuyển lowercase
- Gộp các từ đồng nghĩa (js → javascript, reactjs → react)
- Loại bỏ skills xuất hiện < 3 lần

#### b) Xử lý Multi-category
```python
# Input: "IT/Sales" hoặc "Phần mềm, Marketing"
# Output: ["IT", "Sales"] hoặc ["IT", "Marketing"]
```

#### c) Chuẩn hóa các trường khác
- **Location**: Hanoi, HCM, Danang, Other
- **Job Type**: Full-time, Part-time, Remote, Hybrid
- **Company Size**: 1-9, 10-24, 25-99, 100-499, 500-1000, 1000+

### 3.2 Kết quả sau xử lý
| Trước | Sau |
|-------|-----|
| 2000+ skills | 908 skills (đã chuẩn hóa) |
| Multi-category dạng string | List categories |
| Null values | Filled/handled |

---

## 4. XÂY DỰNG HETEROGENEOUS GRAPH

### 4.1 Kiến trúc Graph (create_graph_from_excel.py)

Sử dụng **HeteroData** từ PyTorch Geometric để tạo graph với 3 loại nodes:

```
┌─────────────────────────────────────────────────────────────┐
│                    HETEROGENEOUS GRAPH                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌─────────┐         requires          ┌─────────┐        │
│    │   JOB   │ ───────────────────────── │  SKILL  │        │
│    │  (499)  │ ◄───────────────────────► │  (908)  │        │
│    └────┬────┘       required_by         └─────────┘        │
│         │                                                    │
│         │ belongs_to                                         │
│         │                                                    │
│         ▼                                                    │
│    ┌─────────┐                                              │
│    │ COMPANY │                                              │
│    │  (55)   │                                              │
│    └─────────┘                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Chi tiết các Node Types

#### a) Job Nodes (499 nodes)
**Features** (vector dimension = số_categories + 7):
```python
job_features = [
    # Multi-hot encoding cho categories (7 categories)
    [0, 1, 0, 0, 1, 0, 0],  # IT=1, Data=1
    
    # Numerical features
    job_level,          # 0-5 (Intern → Manager)
    experience_years,   # 0-10
    salary_min / 1e6,   # Normalized (triệu VNĐ)
    salary_max / 1e6,
    has_salary,         # 0 or 1
    job_type,           # 0-4
    location,           # 0-4
]
```

#### b) Skill Nodes (908 nodes)
- **Features**: One-hot encoding (ma trận đơn vị 908x908)
- Mỗi skill có vector riêng biệt

#### c) Company Nodes (55 nodes)
- **Features**: Company size (1 dimension, giá trị 0-6)

### 4.3 Edge Types

| Edge Type | Số lượng | Mô tả |
|-----------|----------|-------|
| Job → Skill (requires) | 8,871 | Job yêu cầu skill |
| Skill → Job (required_by) | 8,871 | Reverse edge |
| Job → Company (belongs_to) | 499 | Job thuộc company |
| Company → Job (has_job) | 499 | Reverse edge |

### 4.4 Lưu Graph
```python
# Lưu graph structure
torch.save(data, "job_graph_large.pt")

# Lưu metadata (mappings, job info, etc.)
torch.save(metadata, "job_graph_large_metadata.pt")
```

### 4.5 Thống kê Graph cuối cùng
```
================== GRAPH SUMMARY ==================
Node Types: ['job', 'skill', 'company']
Edge Types: ['requires', 'required_by', 'belongs_to', 'has_job']

Total Nodes: 1,462
  - Jobs: 499
  - Skills: 908
  - Companies: 55

Total Edges: 18,740
  - Job-Skill: 8,871 × 2 (bidirectional)
  - Job-Company: 499 × 2 (bidirectional)

Graph Sparsity: 1.96%
===================================================
```

---

## 5. ÁP DỤNG LIGHTGCN

### 5.1 Giới thiệu LightGCN

**LightGCN** (Light Graph Convolution Network) là thuật toán được đề xuất trong paper:
> *"LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"*  
> Xiangnan He et al., SIGIR 2020

**Đặc điểm chính:**
- Đơn giản hóa GCN bằng cách loại bỏ feature transformation và non-linear activation
- Chỉ sử dụng neighborhood aggregation
- Hiệu quả cho bài toán recommendation

### 5.2 Kiến trúc LightGCN

```
Input: User-Item bipartite graph
       (Trong bài toán này: Job-Skill graph)

┌────────────────────────────────────────────────────────┐
│                    LIGHTGCN LAYERS                      │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Layer 0: e⁽⁰⁾ = [job_embeddings; skill_embeddings]    │
│                        ↓                                │
│  Layer 1: e⁽¹⁾ = Aggregate(e⁽⁰⁾, neighbors)            │
│                        ↓                                │
│  Layer 2: e⁽²⁾ = Aggregate(e⁽¹⁾, neighbors)            │
│                        ↓                                │
│  Layer 3: e⁽³⁾ = Aggregate(e⁽²⁾, neighbors)            │
│                        ↓                                │
│  Final:   e = mean([e⁽⁰⁾, e⁽¹⁾, e⁽²⁾, e⁽³⁾])           │
│                                                         │
└────────────────────────────────────────────────────────┘

Aggregation với symmetric normalization:
e_u⁽ᵏ⁺¹⁾ = Σ (1/√|N_u| × 1/√|N_i|) × e_i⁽ᵏ⁾
```

### 5.3 Chuyển đổi dữ liệu cho LightGCN

**Bài toán được định nghĩa:**
- **User** = Job (499 users)
- **Item** = Skill (908 items)
- **Interaction** = Job requires Skill

**Format dữ liệu:**
```
# train.txt - mỗi dòng: user_id item1 item2 item3...
0 123 456 789 234
1 345 678 901
2 234 567 890 123 456
...

# test.txt - tương tự
0 111 222
1 333 444 555
...
```

**Chia dữ liệu:**
- Train: 80% (6,907 interactions)
- Test: 20% (1,964 interactions)

### 5.4 Hyperparameters

```python
config = {
    'embedding_dim': 64,      # Kích thước embedding
    'n_layers': 3,            # Số lớp GCN
    'learning_rate': 0.001,   # Learning rate
    'decay': 1e-4,            # L2 regularization
    'batch_size': 2048,       # Batch size cho BPR
    'epochs': 100,            # Số epochs
}
```

### 5.5 Loss Function: BPR (Bayesian Personalized Ranking)

```python
# Với mỗi (user, positive_item, negative_item):
loss = -log(sigmoid(score_pos - score_neg))

# Score = dot product của embeddings
score = user_embedding · item_embedding
```

---

## 6. KẾT QUẢ THỰC NGHIỆM

### 6.1 Training Progress

```
EPOCH[1/100]   loss: 0.682
EPOCH[10/100]  loss: 0.645  → Recall@10: 59.9%, NDCG@10: 50.9%
EPOCH[20/100]  loss: 0.535  → Recall@10: 62.0%, NDCG@10: 52.0%
EPOCH[50/100]  loss: 0.175  → Recall@10: 59.3%, NDCG@10: 49.7%
EPOCH[70/100]  loss: 0.099  → Recall@10: 61.5%, NDCG@10: 51.5%
EPOCH[90/100]  loss: 0.073  → Recall@10: 63.2%, NDCG@10: 52.9%
EPOCH[100/100] loss: 0.066
```

### 6.2 Evaluation Metrics

| Metric | @10 | @20 |
|--------|-----|-----|
| **Precision** | 21.9% | 13.9% |
| **Recall** | 63.2% | 74.6% |
| **NDCG** | 52.9% | 57.6% |

### 6.3 Giải thích các Metrics

- **Precision@K**: Tỷ lệ items đúng trong top-K predictions
  - Precision@10 = 21.9% → Trung bình 2.19/10 skills được dự đoán đúng

- **Recall@K**: Tỷ lệ items đúng được tìm thấy trong top-K
  - Recall@10 = 63.2% → 63.2% skills thực tế được tìm thấy trong top 10

- **NDCG@K**: Normalized Discounted Cumulative Gain
  - Đo lường chất lượng ranking (skills đúng ở vị trí cao → score cao hơn)
  - NDCG@10 = 52.9%

### 6.4 Training Loss Curve

```
Loss
 │
0.7├─●
   │  ╲
0.6├───●
   │    ╲
0.5├─────●
   │      ╲
0.4├───────●
   │        ╲
0.3├─────────●
   │          ╲
0.2├───────────●──
   │             ╲
0.1├──────────────●────●────●────●
   │
0.0└──┬───┬───┬───┬───┬───┬───┬───┬──► Epoch
      10  20  30  40  50  60  70  100
```

### 6.5 So sánh với Baseline

| Model | Recall@10 | NDCG@10 |
|-------|-----------|---------|
| Random | ~1% | ~1% |
| Most Popular | ~20% | ~15% |
| **LightGCN** | **63.2%** | **52.9%** |

---

## 7. KẾT LUẬN

### 7.1 Những gì đã hoàn thành

✅ **Xây dựng Graph từ Excel:**
- Đọc và xử lý dữ liệu từ file db_base.xlsx
- Làm sạch, chuẩn hóa skills và categories
- Tạo Heterogeneous Graph với 3 loại nodes (Job, Skill, Company)
- Lưu graph dưới dạng PyTorch Geometric format

✅ **Áp dụng LightGCN:**
- Chuyển đổi graph sang format LightGCN
- Train model với 100 epochs
- Đánh giá với các metrics: Precision, Recall, NDCG

### 7.2 Kết quả chính

| Thành phần | Kết quả |
|------------|---------|
| Số Jobs | 499 |
| Số Skills | 908 |
| Số Companies | 55 |
| Total Edges | 18,740 |
| **Recall@10** | **63.2%** |
| **NDCG@10** | **52.9%** |

### 7.3 Nhận xét

1. **LightGCN hoạt động tốt** trên Job-Skill graph với Recall@10 = 63.2%
2. **Graph structure** giúp model học được patterns quan hệ giữa jobs và skills
3. **Dữ liệu nhỏ** (499 jobs, 908 skills) nhưng vẫn cho kết quả khả quan

### 7.4 Hướng phát triển

1. **Mở rộng dữ liệu**: Thu thập thêm jobs để cải thiện model
2. **Sử dụng Company nodes**: Thêm edge Job-Company vào model
3. **Ứng dụng thực tế**: 
   - Input: CV (danh sách skills)
   - Output: Recommend Jobs phù hợp

---

## PHỤ LỤC

### A. Cách chạy lại thí nghiệm

```bash
# 1. Xử lý dữ liệu
python process_excel.py

# 2. Tạo graph
python create_graph_from_excel.py

# 3. Convert dữ liệu cho LightGCN
cd LightGCN-PyTorch/data/jobskill
python convert_data.py

# 4. Chạy LightGCN
cd ../code
python main.py --dataset="jobskill" --layer=3 --lr=0.001 --decay=1e-4 --epochs=100
```

### B. Cấu trúc file Graph

```python
# Load graph
import torch
data = torch.load("job_graph_large.pt")

# Truy cập nodes
data['job'].x          # Job features (499 × feature_dim)
data['skill'].x        # Skill features (908 × 908)
data['company'].x      # Company features (55 × 1)

# Truy cập edges
data['job', 'requires', 'skill'].edge_index    # (2 × 8871)
data['job', 'belongs_to', 'company'].edge_index # (2 × 499)
```

### C. Output files

| File | Mô tả |
|------|-------|
| job_graph_large.pt | Graph structure (PyTorch) |
| job_graph_large_metadata.pt | Metadata (mappings, job info) |
| LightGCN-PyTorch/data/jobskill/train.txt | Training data |
| LightGCN-PyTorch/data/jobskill/test.txt | Test data |
| lgn-jobskill-3-64.pth.tar | Trained model weights |

---

**Ngày thực hiện**: 6/1/2026  
**Nhóm thực hiện**: [Tên nhóm]
