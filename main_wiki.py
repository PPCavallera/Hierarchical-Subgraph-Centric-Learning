from lib.get_data_wiki import get_prepared_dataset, get_raw_data, get_precise_cluster
from lib.models import get_model_1, get_model_2
from lib.training import train_models, get_test_loss, save_loss_graphs
import time

NB_CLUSTER = 4
NB_PAST_VALUE = 8
NB_FUTURE_VALUE = 1

if __name__ == "__main__":
    df = get_raw_data()
    for c in range(NB_CLUSTER):
        df_c = get_precise_cluster(df, NB_CLUSTER, c)
        X_train, y_train, X_test, y_test = get_prepared_dataset(df_c, NB_PAST_VALUE, NB_FUTURE_VALUE, len(df_c.columns))
        
        model_1 = get_model_1(n_features=len(df_c.columns), n_future=NB_FUTURE_VALUE, n_past=NB_PAST_VALUE)
        model_2 = get_model_2(n_features=len(df_c.columns), n_future=NB_FUTURE_VALUE, n_past=NB_PAST_VALUE)
        start_time = time.time()
        model_1, model_2, history_1, history_2 = train_models(X_train, y_train, X_test, y_test, model_1, model_2, bs=32)
        print('TRAINING TIME WIKI CLUSER ' + str(c))
        print("--- %s seconds ---" % (time.time() - start_time))
        save_loss_graphs(history_1, history_2, "loss_wiki_for_c"+str(c))
        get_test_loss(X_test, y_test, model_1, model_2, "wiki_cluster_"+str(c))
