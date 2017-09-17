# real-time-face-recognition-with-facenet

I remember the first day on the job and I was assigned to work on Face Recognition System, but at that time it was like a dream to make a classifier that can do it very well, I was using purely Open-Cv for detection of face and then creating a unique vector for each face. But its accuracy was too less to use it as in any application. Before some months back I read a paper named as  "FaceNet: A Unified Embedding for Face Recognition and Clustering" which present a unified system for face verification.



Facenet is based on learning a Euclidean embedding per image using deep convolution network, Embedding algorithms search for a lot dimensional continuous representation of data. The network is trained such that the squared L2 distances in the embedding space directly correspond to face similarity. Faces of the same person have small distances and faces of distinct people have large distances.



Once this embedding has been produced, then the aforementioned tasks become straight-forward: face verification simply involves thresholding the distance between the two embeddings; recognition becomes a k-NN classification problem, and clustering can be achieved using off-theshelf techniques such as k-means or agglomerative clustering. 

[Complete Post is Here](https://www.linkedin.com/pulse/real-time-face-recognition-using-facenet-ishwar-sawale/)
