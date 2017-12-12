# Step 1:
Modify the value in "Config.py" file: (I have modified it, you can skip this step now)
holding = 100000. The number of documents we will train.
model_path: The path to save the model. You can leave it unchanged or set another name.

# Step 2: Train the document embedding model
Run "python train_doc_fixed.py" to train the document embedding matrix. (Or you can just double click the "train_doc_fixed.py" to start the program. I can do that on my computer.)

## Notice:
There is a parameter called "epochs" in "train_doc_fixed.py" file. It controls how many epochs we will train the model. I don't write code to track how many epochs we have already trained. If you shut down the training process before it completes, make sure to decrease
"epochs" value to how many epochs still left to train (Just roughly, no need to be accurate, in case of you forgot how many epochs we have trained if shutting down the program several times)

# Step 3: Feed the document embedding matrix to a neural network.
You can use IDE to run "logit.py".(May not use terminal because the terminal will close immediately after it prints the accuracy.) There is also a parameter called "epochs". Large epochs lead to high train performance, but maybe over-fitting. Tune the parameter by yourself based on your Machine Learning knowledge!
