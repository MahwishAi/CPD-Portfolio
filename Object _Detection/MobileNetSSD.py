import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model



# Defining variables for parameters for the model    
num_classes = 4
input_size = (224, 224)
learning_rate = 0.001
num_batch = 32  # setting the variable here allows me to easily change it throughout the code if needed
num_epochs = 1  # I have set this to 1 for now as it takes less time to compile and build them model, but i will be training this on 10 epochs later on



base_model = VGG19(include_top=False, 
    weights='imagenet', 
    input_shape=(input_size[0], 
    input_size[1], 3))






base_output = base_model.output
pooled_features = GlobalAveragePooling2D()(base_output)    
hidden_features = Dense(256, activation='relu')(pooled_features)  
predictions = Dense(num_classes, activation='softmax')(hidden_features) 
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False 
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) 



train_data_path = 'train_data'
test_data_path = 'test_data'
train_dataGen = ImageDataGenerator(   
    rescale=1./255,       
    rotation_range=20,   
    width_shift_range=0.2, 
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
    shear_range=0.2,
    zoom_range = 0.2
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




model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // num_batch,
    epochs=num_epochs,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples // num_batch
)




class_labels = list(train_gen.class_indices.keys())
print("Class Labels:", class_labels)
loss, accuracy = model.evaluate(test_gen)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy) 


model.save('ResNet50_Model.h5') 