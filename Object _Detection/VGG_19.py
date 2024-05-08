import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout 
from tensorflow.keras import regularizers

# Set the parameters can adjust these outside of the function for easier modification 
num_classes = 4 #nubeer of classes in the dataset 
input_size = (224, 224) #image input size that the model will process 
learning_rate = 0.001 # can change this to see which learning rate works better 
num_batch = 32 # can adjust the batch size 
num_epochs = 1 #can adjust the epoch, more epoch will take longer to train but will give a better accuracy 


# Load the VGG-19 model
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(input_size[0], input_size[1], 3))


#inplementing  fine turning by only freezing some of the layers 
base_model.summary() # Print the model summary and check the number of trainable parameters and architecture and the layer indicies 
#freezing only some of the layers of the base model
for layer in base_model.layers[:5]: # this will freeze the first 5 layers  
    layer.trainable = False

# allows for the model to leverage the pretrained data for better accuracy 

base_output = base_model.output
pooled_features = GlobalAveragePooling2D()(base_output)

# adding the l2 reuliarizer to the dense layer regularizers.l1 for l1 regularizer and regularizers.l2 for l2 regularizer 
hidden_features = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(pooled_features) # this helps avoid overfitting by penalizing the weights 

#addoing the dropout layer 
hidden_features = Dropout(0.3)(hidden_features) # can change the drop out rate and see which one works best

hidden_features = Dense(256, activation='relu')(pooled_features)
predictions = Dense(num_classes, activation='softmax')(hidden_features)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) #can use different optimizers and see which one works best: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# Prepare the data generators
train_data_path = 'train_data'
test_data_path = 'test_data'

train_dataGen = ImageDataGenerator(   # image data gen is used to augment the data - this helps me create a better dataset to train the model with and help avoid overfitting and better accuracy 
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2
)

train_gen = train_dataGen.flow_from_directory(
    train_data_path,
    target_size=input_size,
    batch_size=num_batch,
    class_mode='categorical',
    subset='training'
)

validation_gen = train_dataGen.flow_from_directory(
    train_data_path,
    target_size=input_size,
    batch_size=num_batch,
    class_mode='categorical',
    subset='validation'
)

test_dataGen = ImageDataGenerator(rescale=1./255)
test_gen = test_dataGen.flow_from_directory(
    test_data_path,
    target_size=input_size,
    batch_size=num_batch,
    class_mode='categorical'
)

# Get class labels
class_labels = list(train_gen.class_indices.keys())
print("Class Labels:", class_labels)

# Train the model
model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // num_batch,
    epochs=num_epochs,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples // num_batch
)

# Evaluate the model
loss, accuracy = model.evaluate(test_gen)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

model.save('VGG19_model.h5') 