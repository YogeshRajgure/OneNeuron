import pandas as pd

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot

def main(data, eta, epochs, filename):

    df = pd.DataFrame(data)

    ####################################

    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=str(filename)+".model")

    save_plot(df, file_name=str(filename)+".png",model_=model)



if __name__ == '__main__': #<< entry point

    OR = {
            "x1" : [0,0,1,1],
            "x2" : [0,1,0,1],
            "y"  : [0,1,1,1],
        }
    
    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    main(data=OR, eta=ETA, epochs=EPOCHS, filename="or")


