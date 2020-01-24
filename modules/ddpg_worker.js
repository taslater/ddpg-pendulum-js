this.window = this

importScripts(
  "./replay.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js",
  "./nn_models.js"
)

tf.enableProdMode()
// tf.setBackend("cpu")

let global,
  state_len,
  replay_buffer,
  //
  trainingActor,
  targetActor,
  trainingCritic,
  targetCritic,
  valueCritic,
  //
  actor_optimizer,
  critic_optimizer,
  value_optimizer,
  //
  actorWeights = [],
  criticWeights = [],
  valueWeights = [],
  //
  actorTau,
  criticTau,
  //
  stateBuffer,
  stateView

let ready = false,
  training = false,
  criticUpdatesSinceActorUpdate = 0

onmessage = async e => {
  const msg = e.data
  if (msg.hasOwnProperty("settings")) {
    initialize(
      msg.settings.state_len,
      msg.settings.global,
      msg.settings.actorWeights
    )
  } else if (ready && msg.hasOwnProperty("experience")) {
    // setTimeout(() => {
    replay_buffer.add(msg.experience)
    // }, 20)
    // let action = 0
    if (!training && replay_buffer.data.length > global.steps_before_training) {
      training = true
      train()
    }
  }
}

function initialize(_state_len, _global, _actorWeights) {
  global = _global
  state_len = _state_len

  replay_buffer = new ReplayBuffer(state_len, global)

  targetActor = Actor(false, state_len)
  trainingActor = Actor(true, state_len)
  targetCritic = Critic(false, state_len)
  trainingCritic = Critic(true, state_len)
  valueCritic = ValueCritic(true, state_len)

  tf.tidy(() => {
    const wts = trainingActor.getWeights()
    const new_wts = _actorWeights
    for (let i = 0; i < wts.length; i++) {
      wts[i] = tf.tensor(new_wts[i], wts[i].shape)
    }
    targetActor.setWeights(wts)
    trainingActor.setWeights(wts)
  })
  targetCritic.setWeights(trainingCritic.getWeights())

  actor_optimizer = tf.train.adam(0.001)
  critic_optimizer = tf.train.adam(0.001)
  value_optimizer = tf.train.adam(0.001)

  for (let i = 0; i < trainingActor.trainableWeights.length; i++) {
    actorWeights.push(trainingActor.trainableWeights[i].val)
  }
  for (let i = 0; i < trainingCritic.trainableWeights.length; i++) {
    criticWeights.push(trainingCritic.trainableWeights[i].val)
  }
  for (let i = 0; i < valueCritic.trainableWeights.length; i++) {
    valueWeights.push(valueCritic.trainableWeights[i].val)
  }

  actorTau = global.actorTauInitial
  criticTau = global.criticTauInitial

  ready = true
}

async function train() {
  // console.log(tf.memory().numTensors)
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

  tf.tidy(() => {
    const grads = critic_optimizer.computeGradients(() => {
      return trainingCritic
        .predict(
          [
            mb_s0.add(
              tf.randomNormal(mb_s0.shape, 0, global.training_critic_obs_noise)
            ),
            mb_actions
          ],
          {
            batchSize: global.mb_len
          }
        )
        .squaredDifference(
          targetCritic
            .predict(
              [
                mb_s1.add(
                  tf.randomNormal(
                    mb_s1.shape,
                    0,
                    global.target_critic_obs_noise
                  )
                ),
                targetActor.predict(
                  mb_s1.add(
                    tf.randomNormal(
                      mb_s1.shape,
                      0,
                      global.target_actor_obs_noise
                    )
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
            .mul(tf.scalar(global.discount))
            .add(mb_rewards)
        )
        .sum()
    }, criticWeights).grads
    for (let i = 0; i < grads.length; i++) {
      grads[i] = grads[i].clipByValue(-1, 1)
    }
    critic_optimizer.applyGradients(grads)
  })

  updateCriticWeights()

  // tf.tidy(() => {
  //   const grads = value_optimizer.computeGradients(() => {
  //     return targetCritic
  //       .predict([mb_s0, mb_actions], {
  //         batchSize: global.mb_len
  //       })
  //       .squaredDifference(
  //         valueCritic.predict(mb_s0, {
  //           batchSize: global.mb_len
  //         })
  //       )
  //       .sum()
  //   }, valueWeights).grads
  //   for (let i = 0; i < grads.length; i++) {
  //     grads[i] = grads[i].clipByValue(-1, 1)
  //   }
  //   value_optimizer.applyGradients(grads)
  // })

  if (criticUpdatesSinceActorUpdate > 3) {
    tf.tidy(() => {
      const grads = actor_optimizer.computeGradients(() => {
        return (
          targetCritic
            .apply(
              [
                mb_s1.add(
                  tf.randomNormal(
                    mb_s0.shape,
                    0,
                    global.target_critic_obs_noise
                  )
                ),
                trainingActor.predict(
                  mb_s1.add(
                    tf.randomNormal(
                      mb_s0.shape,
                      0,
                      global.training_actor_obs_noise
                    )
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
            // .sub(valueCritic.predict(mb_s1), {
            //   batchSize: global.mb_len
            // })
            .sum()
            .mul(tf.scalar(-1))
        )
      }, actorWeights).grads
      for (let i = 0; i < grads.length; i++) {
        grads[i] = grads[i].clipByValue(-1, 1)
      }
      actor_optimizer.applyGradients(grads)
    })

    updateActorWeights()

    postMessage({
      newActorWts: targetActor.getWeights().map(t => t.dataSync())
    })
  }

  tf.dispose(mb_s0)
  tf.dispose(mb_actions)
  tf.dispose(mb_rewards)
  tf.dispose(mb_s1)

  decayTau()

  // setTimeout(() => {

  training = false
  // train()
  // }, 0)
}

function decayTau() {
  actorTau *= global.tauDecay
  criticTau *= global.tauDecay
  if (actorTau < global.actorTauMin) {
    actorTau = global.actorTauMin
  }
  if (criticTau < global.criticTauMin) {
    criticTau = global.criticTauMin
  }
}

function updateCriticWeights() {
  updateWeights(targetCritic, trainingCritic, criticTau)
  criticUpdatesSinceActorUpdate++
}

function updateActorWeights() {
  updateWeights(targetActor, trainingActor, actorTau)
  criticUpdatesSinceActorUpdate = 0
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
