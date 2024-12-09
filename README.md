# bttai-humans-vs-llms

# Dataset
arena-human-preference-55k

# Project Overview
What do humans ask LLMs? We will explore a popular benchmark of
human-LLM dialogues, Chatbot Arena, to answer this question. We will use
various exploratory ML methods to partition the dialogues into semantic clusters
and visualize strengths of small vs big and proprietary vs open-source LLMs over
the dialogue topics. 
Dataset: Used the 55k hugging face humans vs llms dataset

# Objectives and goals
- Process and embed data with sentence transformer

- Cluster with k-means with or without dim-reduction; obtain 2D visualization

- BERTopic + Hierarchical Clustering (HDBSCAN)

- Research Questions

- Takeaways

# Methodology
* Performed K-Means Clustering with and without dimension reduction. Used 5D and 2D for Dimension reduction.

* Eliminated prompt rows with models of same category (small vs small, medium vs medium, large vs large) for analysis.
* Performed 2D and 5D cluster analysis of win/loss rate of small, medium and large models.
* Calculated the number of clusters (topics) won by each model category.
* Analyzed the strengths of large models. 
* Explored if there are any particular topics where humans are more likely to pick a winner/loser and if there are topics where they are more uncertain while deciding the result.
   - Analyzed the win/loss ratio and tie ratio of models across clusters.
   - Basis: Humans are more likely to pick a winner/loser in clusters with a higher win/loss ratio and a lower tie ratio.
   - Picked a win/loss ratio threshold of 0.37 and above (for more likely) and 0.37 and below (for less likely) and a tie threshold ratio of 0.5 and below (for more likely) and 0.5 and above (for less likely).
   - Analyzed the common topics across 5D and 2D where humans are more certain in deciding the winner/loser.


# Results and Key Findings
Clustering Results
At first glance, the 2D visualization appeared to show better separation between clusters compared to 5D. However, interpreting clustering performance only based on plots can be misleading, as visualizations don't always capture the underlying coherence of the clusters. => used a similarity report. 
The results was opposite: 5d score was much better than the 2d => 5d map produces better clustering results => highlights the importance of quantitative metrics over visual interpretation in assessing clustering performance.

Large Model Strengths
- Medium Models win/tie in most clusters across 2D and 5D.
- Large Models win/tie in 10 clusters in 2D and in 8 clusters in 5D
   Large Models tend to win in 2 major cluster topics across 2D and 5D: Programming and Creative Writing
- Small Models have a very low win/tie rate across 5D and 2D

Any particular topics where humans are more likely to pick a winner/loser?
- 2D: 15 clusters with major topics namely Programming, Mathematics and Creative Writing.
- 5D: 18 clusters with major topics namely Programming, Humor, Creative Writing and Mathematics. 
To summarize, Humans have a lot more certainty while deciding a winner/loser in common topics like Programming, Mathematics and Creative Writing.

Any particular topics where humans are less likely to pick a winner/loser?
- 2D: 3 clusters with topics under Repetitive instructions.
- 5D: 4 clusters with topics under Humor.

# Visualizations

Clustering

<img width="594" alt="Screenshot 2024-12-08 at 7 41 50 PM" src="https://github.com/user-attachments/assets/21d6acc9-e87c-4d67-934c-1c5789dcccc0">

Clusters won by each model category
<img width="629" alt="Screenshot 2024-12-08 at 7 42 45 PM" src="https://github.com/user-attachments/assets/66a650fd-84b4-4e4b-8a68-d2e5bdebca25">

Clusters with clear winners/losers
<img width="646" alt="Screenshot 2024-12-08 at 7 43 09 PM" src="https://github.com/user-attachments/assets/ffbbc241-25ee-4a78-90fa-b2a8591499ff">

Clusters with uncertain winners/losers
<img width="627" alt="Screenshot 2024-12-08 at 7 43 30 PM" src="https://github.com/user-attachments/assets/fd973834-5835-43c9-82a0-ab79cc860bf1">

# Potential Next Steps

* Define the topic categories ahead of topic labeling 
  - One prompt can contain multiple topic tags
* Improve the accuracy of Topic Modeling
  - Combine different approaches including Supervised/Unsupervised learning method
  - BERTopic, K-means, Decision Tree, Neural Network 
* Develop a method/framework to evaluate topic labeling
  - Semantic similarity is more meaningful in lexical level rather than the text
* LLM Routing 
  - Based on the project’s win/lose/tie analysis, we can implement LLM Routing

# Individual Contributions

- K-Means Clustering: Arushi and My 
- BERTopic: Ishita
- UMap/T-SNE: Min and Tanvi
  
# Prerequisites
Install the following:
Python 3.8 or above
Git
pip (Python package manager)
Virtual environment manager (optional but recommended)

# Clone the Repository:
git clone https://github.com/arushi32588/bttai-humans-vs-llms
cd bttai-humans-vs-llms

# Installation Instructions
Step 1: Set Up a Virtual Environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

# Setting Up the Project Locally
Step 1: Prepare the Dataset
Place the pre-processed embeddings in the dataset.py.
Update the configuration file (config.json) with the correct path to the dataset and output directories.

# Training the Model
Step 1: Run the Training Script
python main.pynb --config config.json
Step 2: Monitor the Logs
The training script generates logs in the logs/ directory. Check the logs for details about model training progress.

# Evaluating Model Performance
Step 1: Generate Clusters
Evaluate the printed clusters

Step 2: Visualize the Clusters
Analyze the clustering visualization results 
