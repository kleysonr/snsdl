import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from snsdl.keras.wrappers.base_model import BaseModel

class ShallowNet(BaseModel):

    def create_model(self,input_shape=None, num_classes=None, optimizer=None):

        optimizers = {
            'adadelta': keras.optimizers.Adadelta()
        }

        # CNN Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compile model
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers[optimizer], metrics=['accuracy'])

        return model