# Graph-Based Analysis and Drop-out Prediction on a MOOC Platform

## 1. Project Objective
- Build a graph-based model to analyze learner behavior on a MOOC platform.
- Predict which learners are at risk of dropping out of a course.
- Address business challenges: improve course completion rates and learner engagement.

## 2. Dataset and Data Model
- **Dataset:** Stanford SNAP MOOC User Action Dataset
- **Nodes:** Users and Course Activities ("targets")
- **Edges:** Actions users take on course activities (e.g., watching a video, completing a quiz, accessing a document)
  - Includes timestamps for each action to model the temporal user journey.
- **Label:** Binary label indicating whether a student drops out after a specific action (target variable for prediction).

## 3. Proposed Solution Architecture
### Phase 1: Exploratory Graph Analysis (using NetworkX)
- **Graph Construction:** Build a directed graph (edge = user's action on a course activity)
- **Pathfinding:** Identify common learning paths; reveal bottlenecks where users get stuck or drop off
- **Centrality Analysis:** Use algorithms (degree, betweenness centrality) to find popular/influential activities
- **Visualization:** Visualize graph subsets to understand user-activity relationships

### Phase 2: Graph-Based Prediction Model (using PyTorch Geometric or DGL)
- **Problem Formulation:** Link prediction problem (predict likelihood of future action/drop-out event)
- **Model Building:**
  - Use GNN to learn node embeddings (users and activities)
  - Embeddings capture node features and network position
  - Link prediction model uses embeddings to predict drop-out likelihood
- **Training & Evaluation:**
  - Train GNN to predict the LABEL column
  - Evaluate accuracy in identifying at-risk users

## 4. Implementation Steps
- **Data Ingestion:**
  - Download `mooc_actions.tsv` and `mooc_action_labels.tsv` from Stanford SNAP
  - Load into Python environment
- **Graph Construction:**
  - Use `USERID` and `TARGETID` columns to build directed graph in Python (NetworkX)
  - Add timestamps as edge attributes
- **Exploratory Analysis:**
  - Use NetworkX for analysis and visualization
- **Data Preparation for GNN:**
  - Prepare data for PyTorch Geometric (PyG) or Deep Graph Library (DGL)
  - Define nodes/edges, create feature tensors, and edge list
- **Model Building:**
  - Build GNN-based link prediction model
  - Start with simple GNN, then explore advanced options (e.g., GraphSAGE)
- **Training & Prediction:**
  - Train model using drop-out LABEL as target
  - Output probability score for each user-activity pair (likelihood of drop-out)
- **Evaluation:**
  - Evaluate model performance using standard ML metrics

## 5. Business Impact
- Success measured by technical metrics and business value
- Model predictions enable early warning system for at-risk learners
- Platform can proactively intervene:
  - Targeted support
  - Personalized content recommendations
  - Nudges to re-engage learners
- Directly improves KPIs:
  - Course Completion Rate
  - Learner Engagement
- Ensures long-term success of the educational platform