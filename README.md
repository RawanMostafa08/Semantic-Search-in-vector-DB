# 🚀 Semantic Search Engine with Vectorized Databases  

## 📚 Introduction  
This project aims to design and implement an **indexing system for a semantic search database** that efficiently retrieves information based on vector space embeddings. The indexing mechanism focuses on a **vector column**, ensuring high accuracy and speed even for large datasets (up to 20 million entries).  

---

## 🌟 What is Semantic Search?  
Semantic search is a technology that enables search engines to understand the **meaning** behind search queries and provide relevant results based on the **context** and **intent** of the user.  

Unlike traditional keyword-based search methods, semantic search uses **natural language processing (NLP)** and **machine learning** to analyze relationships between words, phrases, and concepts.  

For example:  
- **Query**: "What are the best ways to study effectively?"  
- **Result**: Returns tips on studying, time management strategies, and productivity techniques, even if the exact query words are not in the database.  

---

## 📐 Project Scope  
The project implements an indexing system that meets the following requirements:  
- **Data Structure**:  
  - The database contains only two columns:  
    - `ID`: Unique identifier for each row.  
    - `Embedding`: A 70-dimensional vector representing the data.  
- **Indexing**:  
  - Efficiently retrieves the top `k` most similar rows to the input query vector using **cosine similarity**.  
- **Scalability**: Handles datasets with up to **20 million vectors**.  
- **Performance**: Responds in a reasonable time for `k` up to 10.  


## ⚡ Evaluation Criteria  

1. **Accuracy (Recall)**:  
   - The system must accurately retrieve the top `k` most similar vectors for a query.  

2. **Efficiency**:  
   - Efficient retrieval with reasonable memory usage and response time.  

3. **Scalability**:  
   - Handles datasets up to 20 million entries without performance degradation.  
---

## 📈 Performance Highlights  

### Benchmarks  

| **Dataset Size** | **Score** | **Time (s)** | **Peak RAM Usage (MB)** |  
|-------------------|-----------|---------------|--------------------------|  
| 1M               | 0.0       | 1.49          | 8.50                     |  
| 10M              | 0.0       | 4.20          | 22.25                    |  
| 15M              | 0.0       | 5.59          | 11.32                    |  
| 20M              | 0.0       | 6.65          | 3.04                     |  

### Constraints  

| **DB Size**      | **Peak RAM Usage (MB)** | **Time Limit (s)** | **Min Accepted Score** | **Max Index Size (MB)** |  
|-------------------|--------------------------|---------------------|-------------------------|--------------------------|  
| 1M               | 20                       | 3                   | -5000                  | 50                       |  
| 10M              | 50                       | 6                   | -5000                  | 100                      |  
| 15M              | 50                       | 8                   | -5000                  | 150                      |  
| 20M              | 50                       | 10                  | -5000                  | 200                      |  

### Contributors 

<table align="center" >
  <tr>
      <td align="center"><a href="https://github.com/SH8664"><img src="https://avatars.githubusercontent.com/u/113303945?v=4" width="150px;" alt=""/><br /><sub><b>Sara Bisheer</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/rawanMostafa08"><img src="https://avatars.githubusercontent.com/u/97397431?v=4" width="150px;" alt=""/><br /><sub><b>Rawan Mostafa</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com//mennamohamed0207"><img src="https://avatars.githubusercontent.com/u/90017398?v=4" width="150px;" alt=""/><br /><sub><b>Menna Mohammed</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/fatmaebrahim"><img src="https://avatars.githubusercontent.com/u/113191710?v=4" width="150;" alt=""/><br /><sub><b>Fatma Ebrahim</b></sub></a><br /></td>
  </tr>
</table>
