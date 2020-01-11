export function Actor(trainable, state_len) {
  const in_state = tf.input({ shape: [state_len] })
  const dense1 = tf.layers
    .dense({
      units: 32,
      activation: "elu",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-6 }),
      trainable: trainable
    })
    .apply(in_state)
  const dense2 = tf.layers
    .dense({
      units: 32,
      activation: "elu",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-6 }),
      trainable: trainable
    })
    .apply(dense1)
  const dense3 = tf.layers
    .dense({
      units: 32,
      activation: "elu",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-6 }),
      trainable: trainable
    })
    .apply(dense2)
  const out_layer = tf.layers
    .dense({
      units: 1,
      activation: "tanh",
      trainable: trainable
    })
    .apply(dense3)
  return tf.model({
    inputs: in_state,
    outputs: out_layer
  })
}

export function Critic(trainable, state_len) {
  const in_state = tf.input({ shape: [state_len] })
  const in_action = tf.input({ shape: [1] })

  // const concat1 = tf.layers.concatenate().apply([in_state, in_action])
  // "LeakyReLU"
  const dense1 = tf.layers
    .dense({
      units: 128,
      activation: "elu",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-6 }),
      trainable: trainable
    })
    .apply(in_state)
  const concat2 = tf.layers.concatenate().apply([dense1, in_action])
  const dense2 = tf.layers
    .dense({
      units: 128,
      activation: "elu",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-6 }),
      trainable: trainable
    })
    .apply(concat2)
  const dense3 = tf.layers
    .dense({
      units: 128,
      activation: "elu",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-6 }),
      trainable: trainable
    })
    .apply(dense2)
  const dense4 = tf.layers
    .dense({
      units: 128,
      activation: "elu",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-6 }),
      trainable: trainable
    })
    .apply(dense3)
  const out_layer = tf.layers
    .dense({
      units: 1,
      activation: "linear",
      trainable: trainable
    })
    .apply(dense4)
  return tf.model({ inputs: [in_state, in_action], outputs: out_layer })
}
