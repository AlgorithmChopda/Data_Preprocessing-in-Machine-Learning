# Data_Preprocessing-in-Machine-Learning
While working on Images in Machine Learning Projects it is very important to 
Pre-process the images.
Preprocessing the images takes a lot of time and is very error-prone process

we have built and developed the pre-processing Library that pre-process the images 
in this we have :

- read N no of images from N no of directories 
- read the classes of every image
- Convert them to Numpy array
- Normalize-data
- One-hot-encode-data
- reshape-data  [conversion in required dimension]

Just you need to provide the Image directory and the reshape size 



Eg : 

  training_path = '/Kaggle/training_set/'
  testing_path = '/Kaggle/test_set/'
  
  train_obj = DataPreprocessing(training_path, 300)
  train_images, train_labels = train_obj.load_data()

  test_obj = DataPreprocessing(testing_path, 300)
  test_images, test_labels = test_obj.load_data()
