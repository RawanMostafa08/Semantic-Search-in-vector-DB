from typing import Dict, List, Annotated
import numpy as np
import os
import shutil
from sklearn.cluster import MiniBatchKMeans
import pickle
import heapq
import numpy as np
from sklearn.cluster import KMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
ID_SIZE = np.dtype(np.int32).itemsize
DIMENSION = 70

n_clusters = 1000 
batch_size = 1000
nprobe = 15


class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", cluster_dir_path="clusters", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.cluster_dir_paths = cluster_dir_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

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


    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    # def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
    #     scores = []
    #     num_records = self._get_num_records()
    #     # here we assume that the row number is the ID of each vector
    #     for row_num in range(num_records):
    #         vector = self.get_one_row(row_num)
    #         score = self._cal_score(query, vector)
    #         scores.append((score, row_num))
    #     # here we assume that if two rows have the same score, return the lowest ID
    #     scores = sorted(scores, reverse=True)[:top_k]
    #     return [s[1] for s in scores]

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):

        if not os.path.exists(self.cluster_dir_paths):
            raise FileNotFoundError(f"Cluster directory '{self.cluster_dir_paths}' not found")
        
        with open(self.index_path, 'rb') as index_file:
            kmeans = pickle.load(index_file)

        clusters_distance={} #{"id":distance}
        centroids = kmeans.cluster_centers_ 
        labels = kmeans.labels_


        for i in range(0,nprobe):
            distance = self._cal_score(query,centroids[i])
            clusters_distance[int(labels[i])] = distance

        closest_clusters = dict(sorted(clusters_distance.items(), key=lambda item: item[1]))

        closest_cluster_ids = list(closest_clusters.keys())
        # print(closest_cluster_ids)

        nearest_neighbors = []

        for cluster_file in os.listdir(self.cluster_dir_paths):

            if int(cluster_file.split(".")[0]) in closest_cluster_ids: 
                cluster_file_path = os.path.join(self.cluster_dir_paths, cluster_file)

                if not os.path.exists(cluster_file_path):
                    continue 

                with open(cluster_file_path, 'rb') as f:
                    while True:
                        id_bytes = f.read(ID_SIZE)

                        if not id_bytes:
                            break

                        id = np.frombuffer(id_bytes, dtype=np.int32)[0]
                        vector = self.get_one_row(id)

                        distance = self._cal_score(query, vector)
                        # normal list
                        # nearest_neighbors.append((id, vector, distance))

                        # heap queue
                        if len(nearest_neighbors) < top_k:
                            heapq.heappush(nearest_neighbors, (distance,id,vector))
                        else:
                            heapq.heappushpop(nearest_neighbors, (distance,id,vector))

        # heap queue
        nearest_neighbors = sorted(nearest_neighbors, key=lambda x: -x[0])

        # normal list
        # nearest_neighbors = sorted(nearest_neighbors, key=lambda x: -x[2])[:top_k]


        # Return the top-k vectors
        ids = [int(id) for _,id,_ in nearest_neighbors]
        # print("OUR IDs",ids)
        return ids
        
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Placeholder for index building logic

        # 1 000 000 / 4 000 rows = 250 cluster
        # 10 000 000/ 4 000 rows = 2500 cluster
        # 15 000 000/ 4 000 rows = 3750 cluster
        # 20 000 000/ 4 000 rows = 5000 cluster


        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=DB_SEED_NUMBER, batch_size=batch_size)

        for i in range(0, self._get_num_records(), batch_size):
            batch = self.get_n_rows(i, batch_size)
            kmeans.partial_fit(batch)

        with open(self.index_path, 'wb') as index_file:
            pickle.dump(kmeans, index_file)

        # print("Cluster centers",kmeans.cluster_centers_)
        if os.path.exists(self.cluster_dir_paths):
            shutil.rmtree(self.cluster_dir_paths)

        os.makedirs(self.cluster_dir_paths, exist_ok=True)
    
        cluster_files = {}
        for cluster_id in range(n_clusters):
            file_path = os.path.join(self.cluster_dir_paths, f'{cluster_id}.bin')
            cluster_files[cluster_id] = open(file_path, 'wb')
    
        try:
            for i in range(0, self._get_num_records(), batch_size):
                batch = self.get_n_rows(i, batch_size)
                labels = kmeans.predict(batch)
                ids = range(i, i + batch_size)
    
                for label, vector, id in zip(labels, batch, ids):
                    cluster_files[label].write(id.to_bytes(ID_SIZE, byteorder='little'))
        finally:
            for f in cluster_files.values():
                f.close()

