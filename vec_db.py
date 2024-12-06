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

n_clusters_1 = 1000 
batch_size_1 = 1000
nprobe_1 = 30

n_clusters_2 = 500 
batch_size_2 = 1
nprobe_2 = 250


class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_1_file_path = "index1.dat", cluster_1_dir_path="clusters1", cluster_2_dir_path="clusters2", index_2_file_path="index2.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_1_path = index_1_file_path
        self.index_2_path = index_2_file_path
        self.cluster_1_dir_path = cluster_1_dir_path
        self.cluster_2_dir_path = cluster_2_dir_path
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
        self._build_index_1st_level()
        self._build_index_2nd_level()
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
        self._build_index_1st_level()
        self._build_index_2nd_level()

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


    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):

        if not os.path.exists(self.cluster_1_dir_path):
            raise FileNotFoundError(f"Cluster directory '{self.cluster_1_dir_path}' not found")

        if not os.path.exists(self.cluster_2_dir_path):
            raise FileNotFoundError(f"Cluster directory '{self.cluster_2_dir_path}' not found")
        
        with open(self.index_1_path, 'rb') as index_file:
            kmeans_1st = pickle.load(index_file)

        with open(self.index_2_path, 'rb') as index_file:
            kmeans_2nd = pickle.load(index_file)


        second_level_distances={} #{"id":distance}
        second_level_centroids = kmeans_2nd.cluster_centers_
        second_level_labels = kmeans_2nd.labels_



        for i in range(len(second_level_centroids)):
            distance = self._cal_score(query,second_level_centroids[i])
            second_level_distances[int(second_level_labels[i])] = distance

        sorted_second_level = dict(sorted(second_level_distances.items(), key=lambda item: item[1]))

        closest_second_level_clusters = list(sorted_second_level.keys())[:nprobe_2]


        first_level_distances = {}
        first_level_centroids = kmeans_1st.cluster_centers_

        


        nearest_neighbors = []

        for cluster_file in closest_second_level_clusters:

            cluster_file_path = os.path.join(self.cluster_1_dir_path, f"{cluster_file}.bin")

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

    def _build_index_1st_level(self):
        # Placeholder for index building logic

        # 1 000 000 / 1 000 rows = 1000 cluster
        # 10 000 000/ 1 000 rows = 10 000 cluster
        # 15 000 000/ 1 000 rows = 15 000 cluster
        # 20 000 000/ 1 000 rows = 20 000 cluster

        kmeans = MiniBatchKMeans(n_clusters=n_clusters_1, random_state=DB_SEED_NUMBER, batch_size=batch_size_1)

        for i in range(0, self._get_num_records(), batch_size_1):
            batch = self.get_n_rows(i, batch_size_1)
            kmeans.partial_fit(batch)

        with open(self.index_1_path, 'wb') as index_file:
            pickle.dump(kmeans, index_file)

        # print("Cluster centers",kmeans.cluster_centers_)
        if os.path.exists(self.cluster_1_dir_path):
            shutil.rmtree(self.cluster_1_dir_path)

        os.makedirs(self.cluster_1_dir_path, exist_ok=True)
    
        cluster_files = {}
        for cluster_id in range(n_clusters_1):
            file_path = os.path.join(self.cluster_1_dir_path, f'{cluster_id}.bin')
            cluster_files[cluster_id] = open(file_path, 'wb')
    
        try:
            for i in range(0, self._get_num_records(), batch_size_1):
                batch = self.get_n_rows(i, batch_size_1)
                labels = kmeans.predict(batch)
                ids = range(i, i + batch_size_1)
    
                for label, id in zip(labels, ids):
                    cluster_files[label].write(id.to_bytes(ID_SIZE, byteorder='little'))
        finally:
            for f in cluster_files.values():
                f.close()



    def _build_index_2nd_level(self):
        kmeans_2nd_level = MiniBatchKMeans(n_clusters=n_clusters_2, random_state=DB_SEED_NUMBER, batch_size=batch_size_2)
        
        with open(self.index_1_path, 'rb') as index_file:
            kmeans_1st_level = pickle.load(index_file)

        centroids = kmeans_1st_level.cluster_centers_ 
        # labels = kmeans_1st_level.labels_

        kmeans_2nd_level.fit(centroids)


        with open(self.index_2_path, 'wb') as index_file:
            pickle.dump(kmeans_2nd_level, index_file)


        if os.path.exists(self.cluster_2_dir_path):
            shutil.rmtree(self.cluster_2_dir_path)

        os.makedirs(self.cluster_2_dir_path, exist_ok=True)


        # try:
        #     labels_2nd_level = kmeans_2nd_level.predict(centroids)

        #     cluster_files = {}
        #     for cluster_id in range(n_clusters_2):
        #         file_path = os.path.join(self.cluster_dir_2_paths, f'{cluster_id}.bin')
        #         cluster_files[cluster_id] = open(file_path, 'wb')
    

        #         for label, id in zip(labels, ids):
        #             cluster_files[label].write(id.to_bytes(ID_SIZE, byteorder='little'))

        # finally:
        #     for f in cluster_files.values():
        #         f.close()

        # Predict second-level clusters for the first-level centroids
        second_level_labels = kmeans_2nd_level.predict(centroids)

        # Group first-level cluster IDs by second-level cluster
        cluster_mapping = {cluster_id: [] for cluster_id in range(n_clusters_2)}
        for first_level_id, second_level_id in enumerate(second_level_labels):
            cluster_mapping[second_level_id].append(first_level_id)

        # Write each second-level cluster's associated first-level cluster IDs to files
        for cluster_id, first_level_ids in cluster_mapping.items():
            file_path = os.path.join(self.cluster_2_dir_path, f'{cluster_id}.bin')
            with open(file_path, 'wb') as file:
                for first_level_id in first_level_ids:
                    file.write(first_level_id.to_bytes(ID_SIZE, byteorder='little'))

