from typing import Dict, List, Annotated
import numpy as np
import os
import shutil
from sklearn.cluster import MiniBatchKMeans
import pickle
import heapq
import numpy as np

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat",index_file_path="clusters1", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_file = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.vectors = self.generate_database(db_size)

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()
        return vectors

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_n_rows(self, row_num: int, n: int) -> np.ndarray:
        # This function loads a specified number of rows starting from row_num
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(n, DIMENSION), offset=offset)
            return np.array(mmap_vector)
        except Exception as e:
            return f"An error occurred: {e}"


    def get_n_random_rows(self, indices) -> np.ndarray:
        try:
            min_idx = indices.min()
            max_idx = indices.max()
            offset = min_idx * DIMENSION * ELEMENT_SIZE
            n_rows = max_idx - min_idx + 1

            mmap_vector = np.memmap(
                self.db_path,
                dtype=np.float32,
                mode='r',
                shape=(n_rows, DIMENSION),
                offset=offset
            ) 
            relative_indices = indices - min_idx
            return np.array(mmap_vector[relative_indices])
        except Exception as e:
            return np.empty((len(indices), DIMENSION), dtype=np.float32)


    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
        

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):

        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Cluster directory '{self.index_file}' not found")
        
        num_records= self._get_num_records()  
        db_size = num_records//10**6  
        file_path = os.path.join(self.index_file, f'centroids{db_size}.dat')

        with open(file_path, 'rb') as centroids_file:
            centroids_1 = pickle.load(centroids_file)
            
        if num_records == 10**6:
            available_ram = 20*10**6
            scaling_factor = 500
            nprobe_1 = 30
            nprobe_2 = 20
            
        elif num_records == 10*10**6:
            available_ram = 50*10**6
            scaling_factor = 1000
            nprobe_1 = 50
            nprobe_2 = 20

        elif num_records == 15*10**6:
            available_ram = 50*10**6
            scaling_factor = 2000
            nprobe_1 = 50
            nprobe_2 = 20
        
        elif num_records == 20*10**6:
            available_ram = 50*10**6
            scaling_factor = 10000
            nprobe_1 = 20
            nprobe_2 = 10

            
        vector_size_bytes = DIMENSION * ELEMENT_SIZE  
        max_batch_size = available_ram // (vector_size_bytes * scaling_factor)
        
        # First-level clustering
        nearest_neighbors_1 = sorted(
            [(centroid_1, self._cal_score(query, centroid_1),i) for i,centroid_1 in enumerate(centroids_1)],
            key=lambda x: -x[1]
        )[:nprobe_1]
    
        # Second-level clustering
        top_k_heap = []
        for _,_,i in nearest_neighbors_1:

            file_path = os.path.join(self.index_file, f'{i}.dat')
            with open(file_path, 'rb') as index_file:
                index = pickle.load(index_file)
                
            nearest_neighbors_2 = sorted(
                [(centroid_2, self._cal_score(query, centroid_2),i) for i,centroid_2 in enumerate(index.keys())],
                key=lambda x: -x[1]
            )[:nprobe_2]
                
            for centroid_2, _,_ in nearest_neighbors_2: 
                cluster_ids = index[centroid_2]
                
                cluster_ids_np = np.array(cluster_ids)
                batch_size = min(len(cluster_ids), max_batch_size)
                if batch_size == 0:
                    batch_size = 1
                    
                for i in range(0,len(cluster_ids),batch_size):
                    batch_ids = cluster_ids_np[i:i + batch_size]
                    cluster_vectors = self.get_n_random_rows(batch_ids)
                    
                    # print(len(cluster_ids),len(batch_ids)) 
                    
                    for idx, vector in enumerate(cluster_vectors):
                        distance = self._cal_score(query, vector)
                        if len(top_k_heap) < top_k:
                            heapq.heappush(top_k_heap, (distance, batch_ids[idx]))
                        else:
                            heapq.heappushpop(top_k_heap, (distance, batch_ids[idx]))

        
        top_k_results = [heapq.heappop(top_k_heap)[1] for _ in range(len(top_k_heap))]
        top_k_results.reverse() 
    
        return top_k_results
        

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity


    def _build_index(self):
        batch_size_1 = 10
        batch_size_2 = 10
        max_iter = 200
        
        num_records= self._get_num_records() 
        
        if num_records == 10**6:
            n_clusters_1 = int(4 * np.sqrt(num_records))
            
        elif num_records == 10*10**6:
            n_clusters_1 = 7000

        elif num_records == 15*10**6:
            n_clusters_1 = 10000
        
        elif num_records == 20*10**6:
            n_clusters_1 = 13000
            batch_size_1 = 1000
            batch_size_2 = 10
            max_iter = 1000

        # n_clusters_1 = int(4 * np.sqrt(num_records))

                        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters_1, batch_size=batch_size_1, max_iter=max_iter,n_init=10)

        vectors = self.get_all_rows()

        kmeans.fit(vectors)

        labels = kmeans.predict(vectors)
        centroids = kmeans.cluster_centers_

        cluster_mapping = {tuple(centroid): [] for centroid in centroids}

        for vector_id, label in enumerate(labels):
            centroid_key = tuple(centroids[label])
            cluster_mapping[centroid_key].append(vector_id)
        
        min_length=float('inf')
        for centroid,vector_ids in cluster_mapping.items():
            if len(vector_ids) < min_length and len(vector_ids) > 0:
                min_length = len(vector_ids)
        
        n_clusters_2 = min_length

        db_size = num_records//10**6  
        file_path = os.path.join(self.index_file, f'centroids{db_size}.dat')
        
        with open(file_path, 'wb') as centroids_file:
            pickle.dump(centroids, centroids_file)

        ############################## SECOND LEVEL #######################################

        kmeans_2nd_level = MiniBatchKMeans(n_clusters=n_clusters_2, batch_size=batch_size_2, max_iter=200,n_init=10)

        if os.path.exists(self.clusters_dir_path):
            shutil.rmtree(self.clusters_dir_path)

        os.makedirs(self.clusters_dir_path, exist_ok=True)
        
        for i,centroid_1 in enumerate(cluster_mapping.keys()):
        
            cluster_vector_ids = cluster_mapping[centroid_1]
            cluster_vector_ids_np = np.array(cluster_vector_ids)
    
            cluster_vectors = self.get_n_random_rows(cluster_vector_ids_np)
    
            labels = kmeans_2nd_level.fit_predict(cluster_vectors)
            centroids_2 = kmeans_2nd_level.cluster_centers_

    
            cluster_mapping_2 = {tuple(centroid_2): [] for centroid_2 in centroids_2}
    
            for vector_id, label in zip(cluster_vector_ids,labels):
                centroid_2_key = tuple(centroids_2[label])
                cluster_mapping_2[centroid_2_key].append(vector_id)  
                
            file_path = os.path.join(self.index_file, f'{i}.dat')
            with open(file_path, 'wb') as index_file:
                pickle.dump(cluster_mapping_2, index_file)
            