import pandas as pd
import logging
import os

#from utils.model import Perceptron
#using my own created library uploaded on pypi
from oneNeuron.perceptron import Perceptron

#from utils.all_utils import prepare_data, save_model, save_plot
from oneNeuron.all_utils import prepare_data, save_model, save_plot


logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"

os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")


def main(data, eta, epochs, filename):

    df = pd.DataFrame(data)
    logging.info(f"this is the \n{df}")
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

    try:
        logging.info(">>>>>>>>>>>>>>>> Starting training >>>>>>>>>>>>>>>>")
        main(data=OR, eta=ETA, epochs=EPOCHS, filename="or")
        logging.info("<<<<<<<<<<<<<<<< training done successfully <<<<<<<<<<<<\n\n\n\n")
    except Exception as e:
        logging.exception(e)
        raise e


