# Qianfan-VL: Domain-Enhanced General Vision-Language Model Series

Domain Capability Enhancement through Continuous Pre-training | 3B to 70B Parameter Scale | Document Understanding & OCR Enhancement | Reasoning Capability Support

Released August 2025 | Baidu AI Cloud Qianfan Large Model Platform

## Table of Contents

### 1. Core Features
   - 1.1 Multi-Scale Models
   - 1.2 OCR & Document Understanding
   - 1.3 Chain-of-Thought Reasoning

### 2. Architecture & Technical Features
   - 2.1 Overall Architecture
   - 2.2 Technical Innovation
      - Training Pipeline
      - Data Synthesis
      - Kunlun Chip Training

### 3. Scenario Case Studies
   - 3.1 OCR Recognition
   - 3.2 Mathematical Reasoning
   - 3.3 Document Understanding
   - 3.4 Chart Analysis
   - 3.5 Video Understanding

### 4. Quick Start
   - 4.1 Installation
   - 4.2 Example Code
   - 4.3 API Parameters

### 5. Summary

## Core Features

The Qianfan-VL model series is a general-purpose multimodal large model enhanced for enterprise-level multimodal applications. It possesses fundamental general capabilities while offering deep optimization for high-frequency scenarios in industrial deployment. Through three core functions, it precisely meets multimodal understanding needs in different scenarios.

### Multi-Scale Models

Provides 3B, 8B, and 70B model variants to meet different scenario requirements

### OCR & Document Understanding Enhancement

Full-scenario OCR recognition and intelligent understanding capabilities, covering documents, natural scenes, and various application scenarios

### Chain-of-Thought Reasoning Capability

Supports chain-of-thought capabilities, demonstrating excellent performance in complex scenarios like mathematics and reasoning calculations

## Multi-Scale Models Meet Different Scenario Requirements

Provides 3B, 8B, and 70B model variants, allowing enterprises and developers of different scales to find suitable solutions

| Model Name | Context Length | Reasoning Support | Application Scenarios |
|------------|----------------|-------------------|----------------------|
| **Qianfan-VL-3B** | 32k | Not Supported | Edge real-time scenarios, OCR text recognition |
| **Qianfan-VL-8B** | 32k | Supported | Server-side general scenarios, fine-tuning optimization scenarios |
| **Qianfan-VL-70B** | 32k | Supported | Offline data synthesis, complex reasoning computation scenarios |

### General Capability Benchmark Performance

Comprehensive comparison of Qianfan-VL models of all scales with mainstream models on standard multimodal benchmarks

| Benchmark | Qianfan-VL-3B | Qianfan-VL-8B | Qianfan-VL-70B | Intern2.5-VL-8B | Intern2.5-VL-78B | Intern3-VL-8B | Intern3-VL-78B | Qwen2.5-VL-7B | Qwen2.5-VL-72B | GPT4.1 | Claude-Sonnet-3.7 |
|-----------|---------------|---------------|----------------|-----------------|------------------|---------------|----------------|---------------|----------------|--------|-------------------|
| A-Bench_VAL | 74.25 | 75.65 | **79.15** | 76.91 | 76.14 | 75.86 | 75.86 | 76.49 | **79.22** | 70.75 | 67.04 |
| CCBench | 66.27 | 70 | 77.65 | **78.43** | 71.37 | **77.84** | 70.76 | 57.65 | 73.73 | 27.25 | 19.8 |
| SEEDBench_IMG | 75.74 | 78.21 | **78.85** | 77.17 | 77.15 | 77.00 | 77.52 | 76.98 | **78.34** | 69.98 | 71.18 |
| SEEDBench2_Plus | 68.25 | 70.53 | 71.37 | 69.52 | 69.26 | 69.52 | 68.47 | 70.93 | **73.25** | 65.13 | 60.21 |
| MMVet | 47.25 | 54.13 | 56.42 | 65.14 | 69.72 | **80.28** | **78.90** | 70.64 | 75.69 | 56.88 | 69.72 |
| ScienceQA_TEST | 95.24 | **98.17** | **98.17** | 98.07 | 97.72 | 97.97 | 97.17 | 85.47 | 92.51 | 76.2 | 70.65 |
| ScienceQA_VAL | 94.71 | 97.38 | **98.28** | 97.42 | 96.14 | **97.81** | 95.14 | 83.59 | 91.32 | 74.73 | 69.05 |
| MMT-Bench_VAL | 61.21 | 63.83 | 69.17 | 62.49 | 65.14 | 65.17 | 63.67 | 61.40 | **69.49** | 49.25 | 53.95 |
| MTVQA_TEST | 26.95 | 30.17 | 31.03 | 27.71 | 30.70 | 30.30 | 27.62 | 29.08 | **31.48** | 27.15 | **34.95** |
| BLINK | 51.08 | 56.23 | 58.34 | 54.44 | 53.34 | 55.87 | 51.87 | 54.55 | **63.02** | 40.14 | 43.4 |
| MMStar | 59.07 | 67.4 | **69.07** | 62.86 | 64.13 | **68.40** | 66.07 | 61.53 | 66.00 | 45.47 | 51.53 |
| RealWorldQA | 64.97 | 70.98 | 71.76 | 69.41 | **74.12** | 71.11 | **74.25** | 69.28 | 73.86 | 61.31 | 60.65 |
| Q-Bench1_VAL | 72.57 | 75.99 | **78.33** | 73.90 | 74.98 | 75.99 | 77.99 | 78.10 | **79.93** | 70.43 | 74.25 |
| POPE | 85.15 | 86.84 | 88.79 | 89.06 | 87.14 | **90.59** | 88.87 | 85.97 | 83.35 | 82.15 | 82.07 |

## OCR & Document Understanding Enhancement

Focuses on two distinctive capabilities: full-scenario OCR recognition and complex layout document understanding, demonstrating excellent performance in multiple benchmark tests and providing high-precision visual understanding solutions for enterprise-level applications

### Full-Scenario OCR Tasks
- **Handwriting Recognition:** Chinese and English handwriting recognition, supporting various fonts like cursive and regular script
- **Formula Recognition:** Precise mathematical formula recognition and conversion to LaTeX format
- **Natural Scene Text Recognition:** Text detection in complex environments like street views, signs, and markers
- **Card/Document Information Extraction:** Structured information extraction from ID cards, driver's licenses, business licenses, etc.

### Complex Layout Document Understanding
- **Layout Analysis:** Automatic recognition of layout elements like titles, paragraphs, charts, and tables
- **Table Understanding:** Complex table structure parsing, supporting merged cells and multi-level headers
- **Chart Understanding:** Data extraction and analysis of bar charts, line charts, pie charts, etc.
- **Document Q&A:** Intelligent question answering and information retrieval based on document content
- **Document Parsing:** Structured parsing of PDF, Word, and other format documents

### OCR & Document Understanding Benchmark Performance

Comprehensive comparison of Qianfan-VL models of all scales with mainstream models on OCR and document understanding professional benchmarks

| Benchmark | Qianfan-VL-3B | Qianfan-VL-8B | Qianfan-VL-70B | Qwen2.5-VL-3B | Intern2.5-VL-8B | Intern2.5-VL-78B | Intern3-VL-8B | Intern3-VL-78B | Qwen2.5-VL-7B | Qwen2.5-VL-72B | GPT4.1 | Claude-Sonnet-3.7 |
|-----------|---------------|---------------|----------------|---------------|-----------------|------------------|---------------|----------------|---------------|----------------|-------------------|
| OCRBench | 834 | 852 | 847 | 810 | 822 | 822 | **881** | 847 | **883** | 874 | 702 | 731 |
| AI2D_TEST | 81.76 | 85.33 | **86.76** | 77.07 | 84.391 | 83.48 | **85.07** | 83.55 | 80.472 | 83.84 | 65.12 | 60.75 |
| OCRVQA_TEST | 66.5 | 69.14 | **72.56** | 69.24 | 30.85 | 43.32 | 39.03 | 35.58 | **71.02** | 66.8 | 38.28 | 26.79 |
| TextVQA_VAL | 79.67 | 82.42 | 82.97 | 79.09 | 78.96 | **84.27** | 82.15 | 83.52 | **84.962** | 83.26 | 61.37 | 61.2 |
| DocVQA_VAL | 90.47 | **94.13** | 93.63 | 92.71 | 92 | 93.59 | 92.04 | 83.82 | 94.91 | **95.75** | 76.13 | 78.75 |
| ChartQA_TEST | 81.76 | **88** | **88** | 83.4 | 83.32 | 85.32 | 85.76 | 82.04 | 86.68 | 87.16 | 31.2 | 23.96 |
| CharXiv_DQ | 90.88 | **94.98** | **96.38** | 79.22 | 73.85 | 83.92 | 75.7 | 84.9 | 78.6 | 88.48 | 73.7 | 90.7 |
| CharXiv_RQ | 66.9 | **90.8** | **97.4** | 33 | 37.4 | 40.9 | 40.2 | 40.3 | 44.2 | 50.1 | 46.4 | 68.4 |

## Chain-of-Thought Reasoning Capability

8B and 70B models support chain-of-thought capability activation through special tokens, covering complex chart understanding, visual reasoning, mathematical problem-solving, and more scenarios. These tasks typically require combinatorial reasoning based on visual information and external knowledge. We synthesized extensive visual/textual reasoning data and integrated it into Qianfan-VL's post-training, significantly improving performance on reasoning and computation-related tasks as shown by benchmark results.

### Core Reasoning Application Scenarios

#### Complex Chart Understanding & Reasoning
- **Data Analysis:** Extract key information from complex charts for reasoning analysis
- **Trend Prediction:** Trend judgment and prediction based on historical data charts
- **Correlation Reasoning:** Cross-analysis and correlation reasoning of multi-chart data
- **Statistical Computation:** Statistical analysis and quantitative calculation of chart data

#### Mathematical Problem-Solving & Visual Reasoning
- **Geometric Reasoning:** Spatial figure relationship understanding and theorem application
- **Formula Recognition:** Precise recognition and understanding of complex mathematical formulas
- **Step-by-step Solution:** Clear problem-solving process and step presentation
- **Logical Inference:** Logic reasoning and problem-solving based on visual cues

### Mathematical Problem-Solving Benchmark Performance

| Benchmark | Qianfan-VL-8B | Qianfan-VL-70B | Intern2.5-VL-8B | Intern2.5-VL-78B | Intern3-VL-8B | Intern3-VL-78B | Qwen2.5-VL-7B | Qwen2.5-VL-72B | GPT4.1 | Claude-Sonnet-3.7 |
|-----------|---------------|----------------|-----------------|------------------|---------------|----------------|---------------|----------------|--------|-------------------|
| MathVista-mini | 69.5 | **75.5** | 69.5 | 71.1 | 69.5 | 70.1 | 67.20 | 73.9 | 54.60 | 72.20 |
| MathVision | 27.46 | **45.52** | 21.48 | 33.48 | 29.61 | 34.8 | 25.95 | 39.34 | 34.37 | 43.91 |
| MathVerse | 44.7 | **57.34** | 30.96 | 43.32 | 43.68 | 49.26 | 44.21 | 55.18 | 41.98 | 50.56 |
| ChartQA_Pro | 50.52 | **51.36** | 19.38 | 47.92 | 37.32 | 44.43 | 43.73 | 45.3 | 35.99 | 46.13 |
| HallusionBench | 49.1 | **54.1** | 49.7 | 40.5 | 49.2 | 40.2 | 47.9 | 49.9 | 29.49 | 36.5 |
| InHouse Dataset A | 48.78 | **59.28** | 26 | 43.40 | 40.64 | 41.47 | 45.58 | 57.20 | 21.62 | 42.90 |
| InHouse Dataset B | 61.85 | **65.83** | 26.81 | 39.7 | 36.25 | 42.65 | 30.62 | 59.68 | 15.05 | 24.91 |

## Architecture Design & Technical Features

Through advanced multimodal architecture design and three major technical innovations, Qianfan-VL achieves domain-enhanced general vision-language capabilities

### Overall Architecture

Qianfan-VL adopts advanced multimodal architecture, integrating industry best practices and autonomous innovations

![Qianfan-VL Architecture](images/qianfan_vl_arch_professional.svg)

#### Core Architecture Components

**Language Model**
Based on Llama 3.1 architecture, enhanced through vocabulary expansion and localization with 3T Chinese-English corpus, supporting mixed Chinese-English understanding

**Vision Encoder**
Initialized with InternViT, supporting dynamic patching for different resolution images, with maximum support for 4K resolution input

**Cross-modal Fusion**
MLP adapter achieves seamless bridging between vision and language modalities, ensuring accuracy and efficiency of information transfer

## Technical Innovation & Features

### Capability Enhancement Training Pipeline
Innovative four-stage training strategy that significantly enhances domain capabilities while maintaining general capabilities

### High-Precision Data Synthesis Technology
Combines traditional CV models with programmatic generation to efficiently construct high-quality training data

### Large-Scale Kunlun Chip Training
Completed training entirely using Baidu's self-developed Kunlun P800 chips, demonstrating the mature capabilities of domestic AI infrastructure

## Capability Enhancement Training Pipeline

Innovative four-stage progressive training strategy that significantly enhances domain capabilities while maintaining general capabilities

![Training Pipeline](images/training_pipeline_professional.svg)

**Stage 1: Cross-modal Alignment** - This stage aims to establish basic vision-language connection mapping, using a training strategy that only updates MLP Adapter while freezing Vision Encoder and LLM, trained with 100B tokens of general knowledge data. This stage is necessary, otherwise it will affect overall performance.

**Stage 2: General Knowledge Injection** - Focusing on the amount of injected data, trying to cover all training data, using full-parameter update training strategy with 3.5T tokens of general knowledge data. This stage builds the model's strong foundational capabilities while including sufficient proportion of text corpus to prevent catastrophic forgetting of LLM knowledge.

**Stage 3: Domain-Enhanced Knowledge Injection** - Carefully selecting high-quality data for domains to be enhanced, including task data for enhanced domains while integrating general data sampling to maintain general knowledge and prevent catastrophic forgetting, using full-parameter update training with 300B tokens of domain-specific data and general sampled data. This stage achieves significant enhancement of professional capabilities.

**Stage 4: Post-training** - This stage aims to improve instruction following ability and preference alignment, using full-parameter update training strategy with 1B tokens of instruction fine-tuning data. Uses high-quality alignment data including complex instruction following, writing, Q&A, programming, OCR, information extraction, mathematics, reasoning computation tasks, while incorporating sufficient pure text instruction fine-tuning data to maintain text model capabilities.

## High-Precision Data Synthesis Technology

Constructs a large-scale data synthesis pipeline for multimodal tasks, covering core tasks such as document recognition, mathematical problem solving, chart understanding, table recognition, formula recognition, and natural scene OCR. Through refined pipeline design and intermediate process data construction, it achieves large-scale production of high-quality training data.

### Multi-task Data Synthesis Pipeline

#### Document Recognition OCR Pipeline
**Core Tasks:** Comprehensive analysis, image-to-Markdown, and document Q&A
- **Comprehensive Analysis:** Multi-dimensional analysis integrating layout, category, and content, supporting multiple languages and handwritten scanned documents
- **Image-to-Markdown:** Efficient conversion of single/multi-page documents to structured Markdown
- **Document Q&A:** Deep understanding supporting summarization, reasoning, and multi-turn dialogue
- **Data Sources:** Open source datasets like DocVQA/DocReason25K plus proprietary synthesis and secondary enhancement
- **Robustness Enhancement:** Real noise simulation through bitmap, erosion/dilation, Gaussian blur, etc.

Synthesis Scale: 450M samples | Quality Loop: Multi-VLM cross-validation

#### Mathematical Problem Solving OCR Pipeline
**Core Advantages:** Customized educational data construction + enhanced visual mathematical reasoning
- **Educational Data Preprocessing:** Collect multilingual high-quality problem-solving data, standardize terminology and symbols, structure problems/conditions/steps/formulas, build educational scenario data sources
- **Problem-Solving Data Synthesis:** Combine knowledge systems to synthesize photo problem-solving scenario data through structured expression→LaTeX→HTML→image pipeline
- **Visual Extraction Enhancement:** For complex scenarios like charts, formulas, and geometry, construct high-quality data through formal description languages like Markdown, LaTeX, and Asymptote combined with HTML rendering
- **Diverse Scenario Synthesis:** Data augmentation through multiple handwriting styles and paper background rendering, constructing multi-scenario multi-level data from K-12 to university difficulty
- **Strict Quality Validation:** Validate through rule-based filtering, rejection sampling, multi-model voting, and OCR field-by-field readback to ensure data quality

Synthesis Scale: 85M problems | Coverage: K-12 to university full spectrum | Quality Assurance: Multiple validation mechanisms

#### Chart Understanding Pipeline
**Core Objective:** Automatically generate high-quality chart Q&A pairs covering data retrieval, visual attributes, and computational Q&A
- **Data Expansion:** Open source dataset sampling + Baidu Image Search API expansion + deduplication processing
- **Chart Summary:** Pre-trained VLM generates structured summaries containing visual and numerical information
- **Two-stage Generation:** Generate questions based on summaries → Generate answers based on questions and summaries
- **LaTeX Rendering:** Arxiv data crawling + regex extraction + TexLive re-rendering for precise descriptions
- **Quality Control:** Thinking model quality checking + manual review dual assurance

Synthesis Scale: 180M charts | Question Types: Data retrieval + visual attributes + computational Q&A

#### Table Recognition Pipeline
**Core Tasks:** Table structure recognition and table Q&A
- **Table Structuring:** Precise recovery of image tables to HTML/LaTeX, supporting complex layouts like borderless tables and contract tables
- **Table Q&A:** Numerical computation, comparative analysis, and information retrieval based on table images
- **Content Generation:** Random table structure (3-20 rows/columns) + Faker library/LLM filling + random cell merging
- **Image Rendering:** 50+ professional CSS themes (statistical reports/technical documents) + Jinja2 + KaTeX engine
- **Data Augmentation:** Geometric transforms + color perturbations + blur processing for rich diversity

Synthesis Scale: 120M tables | Data Sources: TabMWP+MMC-Inst+BigDocs+proprietary synthesis

#### Formula Recognition Pipeline
**Core Capabilities:** Integrated symbol recognition + syntax parsing + semantic understanding
- **Symbol Recognition:** Precise recognition of mathematical symbols, Greek letters, and special notations
- **Structure Parsing:** Complex structures like fractions, radicals, superscripts/subscripts, matrices
- **Semantic Understanding:** Association mapping between formula semantics and mathematical concepts
- **Multi-engine Rendering:** MathJax/KaTeX ensuring rendering consistency
- **Handwriting Simulation:** Diverse handwriting characteristics + paper texture + noise interference

Synthesis Scale: 95M formulas | Support: Full coverage of algebra+geometry+calculus+linear algebra

#### Natural Scene OCR Pipeline
**Core Innovation:** Synthtext-pipeline systematic text image synthesis method
- **Background Filtering:** Lightweight OCR model + image type detection to exclude samples with text/non-static content
- **Scene Understanding:** Semantic segmentation model + monocular depth estimation for region division and 3D structure
- **Real Projection:** Plane detection + perspective projection + random text style natural projection
- **Fusion Enhancement:** Poisson fusion ensuring occlusion, shadow, and texture consistency

Synthesis Scale: 320M scenes | Annotation Precision: Character-level + word-level bounding boxes

## Large-Scale Kunlun Chip Parallel Training

Based on Baidu's self-developed Kunlun P800 chips, constructed an industry-leading ultra-large-scale distributed training system, achieving efficient training through innovative parallel strategies and operator optimization

### Cluster Scale and Data Scale
- **Cluster Scale:** 5000+ Kunlun P800 Parallel
- **Training Data Scale:** 3.5T Token Training Data
- **Scaling Efficiency:** 90%+ Large-scale cluster scaling efficiency

### 3D Parallel Training Strategy
Uses a combination of Data Parallelism (DP), Tensor Parallelism (TP), and Pipeline Parallelism (PP), with dynamic load balancing optimizing distribution based on model layer characteristics. Gradient synchronization optimization reduces AllReduce communication time by 60%, combined with ZeRO-3 state sharding technology for memory optimization. Pipeline scheduling uses 1F1B strategy with bubble rate controlled below 5%, sequence dimension partitioning halves long sequence training memory usage, dynamic batching adaptively adjusts batch size based on sequence length, and selective activation recomputation for checkpoint optimization.

### Kunlun Chip Communication-Computation Fusion Technology
**Architecture Advantages:** In the P800 architecture, communication operators and matrix multiplication operators belong to different hardware units, forming a significant difference from traditional GPGPU architecture. In traditional GPU architecture, communication and computation often compete for the same hardware resources, leading to mutual blocking during execution. The P800 architecture achieves true communication-computation parallelism through hardware separation design of dedicated communication processing units and matrix multiplication processing units. This design brings core advantages of resource isolation, where communication operator execution is completely unaffected by matrix multiplication operators, avoiding resource competition in traditional architectures. Meanwhile, through parallel execution mechanisms, data transmission and matrix operations can be performed simultaneously, significantly improving hardware utilization. More importantly, this architecture can use overlap technology to mutually mask communication latency with computation processes.

**GEMM Communication-Computation Fusion Technology:** By establishing additional bypass streams (BypassStream), we can seamlessly integrate communication operators before and after matrix multiplication operations. The core idea of this mechanism is to establish an independent scheduling system, where bypass streams run independently of main computation streams without blocking the main matrix multiplication pipeline. Meanwhile, through data prefetching mechanisms, data communication is initiated in advance to ensure timely arrival of computation-required data. After computation completion, result communication transmission is immediately initiated, forming a complete pipeline.

**Multi-stream Optimization Implementation:** Taking AllGather and matrix multiplication fusion as an example, through fine data chunking strategies, deep fusion of computation and communication is achieved. Traditional methods require completing the entire AllGather operation first, waiting for all data transmission to finish before starting GEMM computation. The fusion method decomposes data into multiple blocks, with each data block immediately starting corresponding computation after communication completion, forming pipeline parallelism. When communication operators prepare atomic data blocks, matrix multiplication can immediately start operations without waiting for all data to be ready, achieving true pipeline parallelism.

## Scenario Case Studies

### OCR Recognition

#### Handwritten Text Recognition
User: Recognize the text in the image
Qianfan-VL: (Chinese handwritten text recognition example with complete output)

#### Scene Text Recognition
User: Please recognize the text in the image
Qianfan-VL: (Natural scene text detection and recognition example)

#### Invoice Information Extraction
User: Extract invoice information
Qianfan-VL:
```json
{
  "invoice_number": "0110020001",
  "invoice_date": "October 2021",
  "total": "1565.05",
  "total_with_tax": "1580.70",
  "items": [
    {"item_name": "Medicine*Angong Niuhuang Pill", "quantity": "1", "amount": "749.11"},
    {"item_name": "Medicine*Yi'anning Pill", "quantity": "1", "amount": "758.71"},
    {"item_name": "Pharmaceutical*Huoxiang Zhengqi Soft Capsule", "quantity": "2", "amount": "57.23"}
  ]
}
```

### Mathematical Reasoning

Detailed geometry problem-solving process and quadratic function problem solving examples (including complete reasoning steps and verification)

### Document Understanding

#### Document Content Understanding
User: Based on the document content, what specific management requirements did the China Banking Regulatory Commission emphasize for commercial real estate credit in the document? How do these requirements affect banks' risk control?
Qianfan-VL: The document proposes four core requirements: 1) Strictly manage real estate development loans, limiting fund flows to non-compliant projects; 2) Regulate land reserve loans, limiting loan ratios and terms; 3) Strengthen housing consumer loan management, implementing differentiated down payment ratios and interest rates; 4) Strictly control commercial property loans. These requirements effectively reduce banks' credit risks and prevent real estate bubbles by limiting excessive fund flows into real estate, raising loan thresholds, and implementing differentiated pricing.

#### Table Structure Understanding
Support for complex table HTML parsing and structured extraction

### Chart Analysis

#### Chart Data Analysis
Trend analysis, data extraction, and reasoning judgment based on chart content

#### Stock Trend Analysis
Professional analysis and trend description of financial charts such as the Shanghai Composite Index

### Video Understanding
Description and understanding of video content, including scene recognition, action understanding, etc.

## Quick Start

### Installation & Configuration

Install the OpenAI SDK to use the Qianfan-VL series models on the Qianfan large model platform:

```bash
pip install openai
```

### Functional Example Code

#### OCR Text Recognition

```python
from openai import OpenAI
import base64
import os

client = OpenAI(
    api_key = os.environ.get("QIANFAN_SECRET_KEY"),
    base_url = "https://qianfan.baidubce.com/v2",
    default_headers = {"X-Access-Key": os.environ.get("QIANFAN_ACCESS_KEY")}
)

# Read and encode image
with open("document.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# OCR recognition
response = client.chat.completions.create(
    model = "Qianfan-VL-8B",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please recognize all text in the image, maintaining original format"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ],
    temperature = 0.1,  # Low temperature for accuracy
    max_tokens = 2048
)

print(response.choices[0].message.content)
```

#### Mathematical Problem Solving

```python
from openai import OpenAI
import base64
import os

client = OpenAI(
    api_key = os.environ.get("QIANFAN_SECRET_KEY"),
    base_url = "https://qianfan.baidubce.com/v2",
    default_headers = {"X-Access-Key": os.environ.get("QIANFAN_ACCESS_KEY")}
)

# Read math problem image
with open("math_problem.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Activate chain-of-thought reasoning for solving
response = client.chat.completions.create(
    model = "Qianfan-VL-70B",  # Use large model for stronger reasoning capability
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": """Please solve this math problem:
1. Explain the solution approach in detail
2. Provide step-by-step derivation process
3. Use LaTeX format for mathematical formulas
4. Provide the final answer"""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ],
    temperature = 0.2,
    max_tokens = 4096
)

print(response.choices[0].message.content)
```

#### Document Understanding & Extraction

```python
from openai import OpenAI
import base64
import os

client = OpenAI(
    api_key = os.environ.get("QIANFAN_SECRET_KEY"),
    base_url = "https://qianfan.baidubce.com/v2",
    default_headers = {"X-Access-Key": os.environ.get("QIANFAN_ACCESS_KEY")}
)

# Read document image
with open("contract.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Document understanding and information extraction
response = client.chat.completions.create(
    model = "Qianfan-VL-8B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this contract document:
1. Identify document type and key clauses
2. Extract information about both parties
3. Mark important dates and amounts
4. Identify potential risk clauses
5. Output in JSON format"""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ],
    temperature = 0.1,
    max_tokens = 3000
)

import json
result = json.loads(response.choices[0].message.content)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

#### Chain-of-Thought Reasoning Capability

```python
from openai import OpenAI
import base64
import os

client = OpenAI(
    api_key = os.environ.get("QIANFAN_SECRET_KEY"),
    base_url = "https://qianfan.baidubce.com/v2",
    default_headers = {"X-Access-Key": os.environ.get("QIANFAN_ACCESS_KEY")}
)

# Read complex chart
with open("complex_chart.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Use chain-of-thought for deep analysis
response = client.chat.completions.create(
    model = "Qianfan-VL-70B",
    messages=[
        {
            "role": "system",
            "content": "You are a data analysis expert, please use chain-of-thought methodology for analysis"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this chart and answer:
1. What is the overall trend of the data?
2. What anomalies need attention?
3. Based on current data, what are the predictions for the next 3 months?
4. Provide your reasoning process and evidence"""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ],
    temperature = 0.3,
    max_tokens = 2048
)

print(response.choices[0].message.content)
```

### API Parameter Description

For detailed API parameter descriptions and calling documentation, please refer to: [Qianfan ModelBuilder API Documentation](https://cloud.baidu.com/doc/qianfan-docs/s/Fm9l6ocai)

## Summary

**Qianfan-VL is positioned as a domain-enhanced general multimodal large language model**, offering multiple specifications of 3B, 8B, and 70B, achieving multi-scale and full-scenario application coverage. Focusing on B2B customer needs, it significantly enhances multiple task capabilities in intelligent office and K-12 education scenarios, including OCR recognition, document parsing, photo problem-solving, chart understanding, and complex table parsing. For scenarios requiring complex reasoning, the thinking capability can be enabled on 8B and 70B models to further enhance model performance.

**On the technical level, it adopts multi-stage progressive continuous pre-training technology**, continuously enhancing the proportion of domain-specific data while maintaining general capabilities, thereby achieving significant improvement in domain capabilities. Based on traditional small models and programmatic synthesis methods, the Qianfan-VL team has constructed a large amount of high-precision training data, significantly increasing data density for long-tail scenarios and improving model generalization. All model sizes were completed through large-scale parallel training powered by 5000+ Kunlun chips, and these models can perform efficient inference on Kunlun chips, GPUs, and other processors.

**The Qianfan-VL series models demonstrate good generalizability among models of the same parameter size**, with excellent performance on specialized domain benchmarks and even better performance on real business benchmarks. Through the domain enhancement technology route, Qianfan-VL provides high-performance solutions that combine both generalizability and specialization for enterprise-level multimodal AI applications.