from lib.get_data_windmill import get_prepared_dataset, get_raw_data, get_precise_cluster
from lib.models import get_model_1, get_model_2
from lib.training import train_models, get_test_loss, save_loss_graphs
import time

if __name__ == "__main__":
    df = get_raw_data()
    for c in range(1):
        df_c = get_precise_cluster(df, 1, c)
        X_train, y_train, X_test, y_test = get_prepared_dataset(df_c, 8, 1, len(df_c.columns))
        
        model_1 = get_model_1(n_features=len(df_c.columns), n_future=1, n_past=8)
        model_2 = get_model_2(n_features=len(df_c.columns), n_future=1, n_past=8)
        start_time = time.time()

        model_1, model_2, history_1, history_2 = train_models(X_train, y_train, X_test, y_test, model_1, model_2, bs=64)
        print('TRAINING TIME WINDMILL CLUSER ' + str(c))
        print("--- %s seconds ---" % (time.time() - start_time))
        save_loss_graphs(history_1, history_2, "loss_windmill_for_c"+str(c))
        get_test_loss(X_test, y_test, model_1, model_2, "windmill_cluster_"+str(c))
