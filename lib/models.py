import tensorflow as tf


def get_model_1(n_past, n_features, n_future, units=100):
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    encoder_l1 = tf.keras.layers.LSTM(units, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)

    encoder_states1 = encoder_outputs1[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(
        n_future)(encoder_outputs1[0])

    decoder_l1 = tf.keras.layers.LSTM(
        units, return_sequences=True)(
        decoder_inputs, initial_state=encoder_states1)
    decoder_outputs1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features))(decoder_l1)

    model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)

    return model_e1d1


def get_model_2(n_past, n_features, n_future, units=100):
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    encoder_l1 = tf.keras.layers.LSTM(
        units, return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = tf.keras.layers.LSTM(units, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(
        n_future)(encoder_outputs2[0])

    decoder_l1 = tf.keras.layers.LSTM(
        units, return_sequences=True)(
        decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(
        units, return_sequences=True)(
        decoder_l1, initial_state=encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features))(decoder_l2)

    model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)

    return model_e2d2


if __name__ == "__main__":
    print(get_model_2(1, 1, 1, 100).summary())
