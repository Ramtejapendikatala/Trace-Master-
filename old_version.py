# # # Load FaceNet Model
# # facenet_model = FaceNet()

# # Load InsightFace Model
# # def load_insightface():
# #     print("[INFO] Loading InsightFace model...")
# #     app1 = FaceAnalysis(name='buffalo_l')
# #     app1.prepare(ctx_id=0, det_size=(640, 640))
# #     return app1

# # app1 = load_insightface()

# # # Extract Features using FaceNet
# # def extract_features(img_path):
# #     img = cv2.imread(img_path)
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     img = cv2.resize(img, (160, 160))  # FaceNet expects 160x160 images
# #     img = np.expand_dims(img, axis=0)
# #     features = facenet_model.embeddings(img)
# #     return features.flatten()

# # Extract Face Embeddings using InsightFace
# def extract_face_embedding(image_path, app1):
#     image = cv2.imread(image_path)
#     faces = app1.get(image)
    
#     if len(faces) == 0:
#         return None
#     return faces[0]['embedding']

# # Compare Face Embeddings
# def compare_faces(embedding1, embedding2, threshold=0.60):
#     similarity = 1 - cosine(embedding1, embedding2)
#     return similarity >= threshold, similarity

# # Check if Image Exists in Complaint Folder
# def is_image_in_complaint_folder(suspected_image_path):
#     suspected_embedding = extract_face_embedding(suspected_image_path, app1)
#     if suspected_embedding is not None:
#         print("InsightFace embedding found for the image")
#     else:
#         return False , ""

#     with open(CSV_FILE1, "r", encoding="utf-8") as file:
#         reader = csv.reader(file)
#         for row in reader:
#             complaint_image = row[1]  # Image filename in complaint data
#             complaint_image_path = os.path.join(UPLOAD_FOLDER1, complaint_image)
            
#             if os.path.isfile(complaint_image_path):
#                 complaint_embedding = extract_face_embedding(complaint_image_path, app1)
#                 if complaint_embedding is not None and suspected_embedding is not None:
#                     is_same_person, similarity_insightface = compare_faces(suspected_embedding, complaint_embedding)
#                     print("InsightFace match:", is_same_person)
#                     if is_same_person:
#                         print("cosine : ",similarity_insightface)
#                         return True, row  # Return matched complaint details
                
#                 # # Fallback to FaceNet if InsightFace fails
#                 # suspected_features = extract_features(suspected_image_path)
#                 # complaint_features = extract_features(complaint_image_path)
#                 # similarity_facenet = cosine_similarity([suspected_features], [complaint_features])[0][0]
#                 # print("FaceNet similarity:", similarity_facenet)
                
#                 # if similarity_facenet > 0.8:
#                 #     return True, row
#                 # if similarity_facenet > 0.65:
#                 #     return True, row
    
#     return False, None

# # Check if Image Exists in Suspected Folder
# def is_image_in_suspected_folder(complaint_image_path):
#     complaint_embedding = extract_face_embedding(complaint_image_path, app1)
#     if complaint_embedding is not None:
#         print("InsightFace embedding found for the image")
#     else:
#         return False , ""

#     with open(CSV_FILE2, "r", encoding="utf-8") as file:
#         reader = csv.reader(file)
#         for row in reader:
#             suspected_image = row[1]  # Image filename in suspected data
#             suspected_image_path = os.path.join(UPLOAD_FOLDER2, suspected_image)
            
#             if os.path.isfile(suspected_image_path):
#                 suspected_embedding = extract_face_embedding(suspected_image_path, app1)
#                 if complaint_embedding is not None and suspected_embedding is not None:
#                     is_same_person, similarity_insightface = compare_faces(complaint_embedding, suspected_embedding)
#                     print("InsightFace match:", is_same_person)
#                     if is_same_person:
#                         print("cosine : ",similarity_insightface)
#                         return True, row  # Return matched suspected details
                
#                 # # Fallback to FaceNet if InsightFace fails
#                 # complaint_features = extract_features(complaint_image_path)
#                 # suspected_features = extract_features(suspected_image_path)
#                 # similarity_facenet = cosine_similarity([complaint_features], [suspected_features])[0][0]
#                 # print("FaceNet similarity:", similarity_facenet)
                
#                 # if similarity_facenet > 0.8:
#                 #     return True, row
#                 # if similarity_facenet > 0.70:
#                 #     return True, row
    
#     return False, None
