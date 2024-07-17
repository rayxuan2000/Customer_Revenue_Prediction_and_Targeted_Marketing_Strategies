# Customer_Revenue_Prediction_and_Targeted_Marketing_Strategies

In this project, I'm going to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer. 

## Dataset
The dataset contains JSON format file, which need to be converted into several individual columns. It's available [here](https://www.kaggle.com/competitions/ga-customer-revenue-prediction/data). A general overview of this dataset is described as follows:

- fullVisitorId- A unique identifier for each user of the Google Merchandise Store.
- channelGrouping - The channel via which the user came to the Store.
- date - The date on which the user visited the Store.
- device - The specifications for the device used to access the Store.
- geoNetwork - This section contains information about the geography of the user.
- socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
- totals - This section contains aggregate values across the session.
- trafficSource - This section contains information about the Traffic Source from which the session originated.
- visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you - -should use a combination of fullVisitorId and visitId.
- visitNumber - The session number for this user. If this is the first session, then this is set to 1.
- visitStartTime - The timestamp (expressed as POSIX time).
- hits - This row and nested fields are populated for any and all types of hits. Provides a record of all page visits.
- customDimensions - This section contains any user-level or session-level custom dimensions that are set for a session. This is a repeated field and has an entry for each dimension that is set.
- totals - This set of columns mostly includes high-level aggregate data.

## What is embedding actually
Let's say we have a feature "color" with values: {"red", "blue", "green"}. We can map these values to integer indices: {"red": 0, "blue": 1, "green": 2}.
Code Example
```
import torch
import torch.nn as nn

# Example categorical data (batch size of 4)
cat_data = torch.tensor([0, 1, 2, 1])  # indices for ["red", "blue", "green", "blue"]

# Define embedding layer
vocab_size = 3  # Number of unique values in the categorical feature
embed_dim = 2   # Size of the embedding vector
embedding_layer = nn.Embedding(vocab_size, embed_dim)

# Pass the categorical data through the embedding layer
embeddings = embedding_layer(cat_data)

print("Categorical Data Indices:", cat_data)
print("Embedding Vectors:\n", embeddings)
```
Output
```
Categorical Data Indices: tensor([0, 1, 2, 1])
Embedding Vectors:
 tensor([[ 0.3550, -0.6486],
         [ 0.7117,  0.2379],
         [ 0.9248,  0.4260],
         [ 0.7117,  0.2379]], grad_fn=<EmbeddingBackward>)

```
## Notes
- Use last 2 months data as validation.
- To make it clear, each row in a batch **to the final nerual network** is as follows: [all nummeric values on that row, cat_emb1_output, cat_emb2_output]. The two embeddings is created based on vocabulary number in each categorical column. That is, cat_emb1 = [colA_emb, colB_emb, ...] and cat_emb2 = [colM_emb, colN_emb...].

- The dimension of categorical column i embedding is calculated as follows: min((max_values[i]+1)//2, 50)

- Details about embedding:

label encoding for categorical columns -> divide cat columns into 2 parts based on vocabulary number -> create a container for each cat column i:

```
emb_dims1 = []
emb_dims2 = []
for i in cat_col_labels1:
    emb_dims1.append((max_values[i], min((max_values[i]+1)//2, 50)))
for i in cat_col_labels2:
    emb_dims2.append((max_values[i], min((max_values[i]+1)//2, 50)))
```

->create embedding layers in NN class:
```
self.emb_layers1 = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims1])
self.emb_layers2 = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims2])
```
e.g. if input is
```
emb_dims1 = [(10, 3), (20, 5), (30, 2)]
```
then output will be 
```
ModuleList(
  (0): Embedding(10, 3)
  (1): Embedding(20, 5)
  (2): Embedding(30, 2)
)
```

-> forward
```
# embedding categorical feature and cat them together
x1 = [emb_layer(cat_data1[:, i].clone().detach()) for i, emb_layer in enumerate(self.emb_layers1)]
# This concatenates along the second dimension (dim=1).
x1 = torch.cat(x1, 1)
```
which is then sent to branch 1. The same goes for branch 2 and goes to bullet point 2 in this section.


## Some interview questions
- What are the key difference between XGBoost and GBM? Improve upon the base gradient boosting machine in the following ways: parallelized implemenatation; regularization and sparsity awareness.
- LightGBM: utilize two novel techniques called Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) which allow the algorithm to run faster. (One side sampling: 保留固定比例的正确和全部错误给下一个weak learner训练)
  
## Code
It has been uploaded to the repo. Be aware that some figures may not display due to environment setting. You could run it on any jupter notebook platform to see.

## Summary
- Conducted target-group based prediction on a 33GB+ Gstore dataset across 58+ fields for potential positive revenue,
presented more actionable operational changes and a better use of marketing budgets.
- Created 32+ comprehensive statistical analysis plots before regression modeling with variable transformations.
- Developed baseline LightGBM model, as well as a Deep Dual-Branch Embedded Neural Network to handle vocabulary-
based categorical feature embedding and numerical data, achieving a 0.014 MSE loss with 92.7% improvement.
- Validated the model’s effectiveness in identifying key customer through lift model analysis with parameterized code.
