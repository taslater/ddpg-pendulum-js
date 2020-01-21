this.window = this

importScripts(
  "./replay.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js",
  "./nn_models.js"
)

tf.enableProdMode()

let global,
  ready = false,
  state_len,
  replay_buffer,
  trainingActor,
  targetActor,
  trainingCritic,
  targetCritic,
  ac_optimizer,
  actorWeights = [],
  actorTau,
  criticTau,
  stateBuffer,
  stateView

onmessage = async e => {
  const msg = e.data
  if (msg.hasOwnProperty("settings")) {
    initialize(msg.settings.state_len, msg.settings.global)
  } else if (msg.hasOwnProperty("experience") && ready) {
    replay_buffer.add(msg.experience)
    let action = 0
    if (replay_buffer.data.length > global.steps_before_training) {
      await train(msg.ep_step)
      action = getAction(msg.experience.s1)
    }
    postMessage({ action: action })
  }
}

function initialize(_state_len, _global) {
  global = _global
  state_len = _state_len

  stateBuffer = new ArrayBuffer(4 * state_len)
  stateView = new Float32Array(stateBuffer)

  replay_buffer = new ReplayBuffer(state_len, global)

  trainingActor = Actor(true, state_len)
  targetActor = Actor(false, state_len)
  trainingCritic = Critic(true, state_len)
  targetCritic = Critic(false, state_len)

  targetActor.setWeights(trainingActor.getWeights())
  targetCritic.setWeights(trainingCritic.getWeights())
  trainingCritic.compile({
    optimizer: tf.train.adam(0.001),
    loss: tf.losses.meanSquaredError
  })

  ac_optimizer = tf.train.adam(0.001)

  for (let i = 0; i < trainingActor.trainableWeights.length; i++) {
    actorWeights.push(trainingActor.trainableWeights[i].val)
  }

  actorTau = global.actorTauInitial
  criticTau = global.criticTauInitial

  ready = true
}

function getAction(state) {
  for (let i = 0; i < state.length; i++) {
    stateView[i] = state[i]
  }
  return tf.tidy(() => {
    return targetActor
      .predict(tf.tensor(stateView, [1, state.length], "float32"), {
        batchSize: 1
      })
      .dataSync()[0]
  })
}

async function train(ep_step) {
  return new Promise((resolve, reject) => {
    replay_buffer.sample()

    const mb_s0 = tf.tensor(
      replay_buffer.mb_s0_view,
      [global.mb_len, state_len],
      "float32"
    )
    const mb_actions = tf.tensor(
      replay_buffer.mb_action_view,
      [global.mb_len, 1],
      "float32"
    )
    const mb_rewards = tf.tensor(
      replay_buffer.mb_reward_view,
      [global.mb_len, 1],
      "float32"
    )
    const mb_s1 = tf.tensor(
      replay_buffer.mb_s1_view,
      [global.mb_len, state_len],
      "float32"
    )

    const pred_next_actions = tf.tidy(() => {
      return targetActor.predict(
        mb_s1.add(tf.randomNormal(mb_s1.shape, 0, global.obs_noise)),
        {
          batchSize: global.mb_len
        }
      )
    })

    const q_pred = tf.tidy(() => {
      return targetCritic
        .predict(
          [
            mb_s1.add(tf.randomNormal(mb_s1.shape, 0, global.obs_noise)),
            pred_next_actions
          ],
          {
            batchSize: global.mb_len
          }
        )
        .mul(tf.scalar(global.discount))
        .add(mb_rewards)
    })

    trainingCritic
      .fit(
        [
          mb_s0.add(tf.randomNormal(mb_s0.shape, 0, global.obs_noise)),
          mb_actions
        ],
        q_pred,
        {
          epochs: 1,
          batchSize: global.mb_len,
          yieldEvery: "never",
          shuffle: true
        }
      )
      .then(() => {
        updateCriticWeights()

        if (ep_step % 2 == 1) {
          tf.tidy(() => {
            const grads = ac_optimizer.computeGradients(() => {
              return targetCritic
                .apply(
                  [
                    mb_s1.add(
                      tf.randomNormal(mb_s1.shape, 0, global.obs_noise)
                    ),
                    trainingActor.predict(
                      mb_s1.add(
                        tf.randomNormal(mb_s1.shape, 0, global.obs_noise)
                      ),
                      {
                        batchSize: global.mb_len
                      }
                    )
                  ],
                  {
                    batchSize: global.mb_len
                  }
                )
                .sum()
                .mul(tf.scalar(-1))
            }, actorWeights).grads
            for (let i = 0; i < grads.length; i++) {
              grads[i] = grads[i].clipByValue(-1, 1)
            }
            ac_optimizer.applyGradients(grads)
          })

          updateActorWeights()
        }

        tf.dispose(mb_s0)
        tf.dispose(mb_actions)
        tf.dispose(mb_rewards)
        tf.dispose(mb_s1)
        tf.dispose(pred_next_actions)
        tf.dispose(q_pred)

        decayTau()

        resolve()
      })
  })
}

function decayTau() {
  actorTau *= global.tauDecay
  criticTau = global.tauDecay
  if (actorTau < global.actorTauMin) {
    actorTau = global.actorTauMin
  }
  if (criticTau < global.criticTauMin) {
    criticTau = global.criticTauMin
  }
}

function updateActorWeights() {
  updateWeights(targetActor, trainingActor, actorTau)
}

function updateCriticWeights() {
  updateWeights(targetCritic, trainingCritic, criticTau)
}

function updateWeights(target_model, training_model, tau) {
  tf.tidy(() => {
    const training_wts = training_model.getWeights()
    const target_wts = target_model.getWeights()
    for (let i = 0; i < training_wts.length; i++) {
      target_wts[i] = target_wts[i]
        .mul(tf.scalar(1 - tau))
        .add(training_wts[i].mul(tf.scalar(tau)))
    }
    target_model.setWeights(target_wts)
  })
}