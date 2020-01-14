import { Actor, Critic } from "./nn_models.js"
import { ReplayBuffer } from "./replay.mjs"
import global from "./parameters.mjs"

tf.enableProdMode()

export class DDPG {
  constructor(state_len) {
    this.state_len = state_len
    this.replay_buffer = new ReplayBuffer()
    this.trainingActor = Actor(true, state_len)
    this.targetActor = Actor(false, state_len)
    this.trainingCritic = Critic(true, state_len)
    this.targetCritic = Critic(false, state_len)
    this.targetActor.setWeights(this.trainingActor.getWeights())
    this.targetCritic.setWeights(this.trainingCritic.getWeights())
    this.trainingCritic.compile({
      optimizer: tf.train.adam(0.001),
      loss: tf.losses.meanSquaredError
    })
    this.actorWeights = []
    for (let i = 0; i < this.trainingActor.trainableWeights.length; i++) {
      this.actorWeights.push(this.trainingActor.trainableWeights[i].val)
    }
    this.ac_optimizer = tf.train.adam(0.001)

    this.obs_noise = 0.01
  }

  train(ep_step) {
    return new Promise((resolve, reject) => {
      // tf.setBackend("webgl")
      const mb = this.replay_buffer.sample().slice()

      const mb_s0 = tf.tensor(
        mb.map(experience => experience.s0),
        [global.mb_len, this.state_len]
      )
      const mb_actions = tf.tensor(
        mb.map(experience => experience.a),
        [global.mb_len, 1]
      )
      const mb_rewards = tf.tensor(
        mb.map(experience => experience.r),
        [global.mb_len, 1]
      )
      const mb_s1 = tf.tensor(
        mb.map(experience => experience.s1),
        [global.mb_len, this.state_len]
      )

      const pred_next_actions = tf.tidy(() => {
        return this.targetActor.predict(
          mb_s1.add(tf.randomNormal(mb_s1.shape, 0, this.obs_noise)),
          {
            batchSize: global.mb_len
          }
        )
      })

      // const q_now = tf.tidy(() => {
      //   return this.targetCritic.predict([mb_s0, mb_actions], {
      //     batchSize: global.mb_len
      //   })
      // })

      const q_pred = tf.tidy(() => {
        return this.targetCritic
          .predict(
            [
              mb_s1.add(tf.randomNormal(mb_s1.shape, 0, this.obs_noise)),
              pred_next_actions
            ],
            {
              batchSize: global.mb_len
            }
          )
          .mul(tf.scalar(global.discount))
          .add(mb_rewards)
        // .sub(q_now)
        // .mul(tf.scalar(global.lr_alpha))
        // .add(q_now)
      })

      this.trainingCritic
        .fit(
          [
            mb_s0.add(tf.randomNormal(mb_s0.shape, 0, this.obs_noise)),
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
          this.updateCriticWeights()

          if (ep_step % 2 == 1) {
            tf.tidy(() => {
              const grads = this.ac_optimizer.computeGradients(() => {
                return this.targetCritic
                  .apply(
                    [
                      mb_s1.add(
                        tf.randomNormal(mb_s1.shape, 0, this.obs_noise)
                      ),
                      this.trainingActor.predict(
                        mb_s1.add(
                          tf.randomNormal(mb_s1.shape, 0, this.obs_noise)
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
              }, this.actorWeights).grads
              for (let i = 0; i < grads.length; i++) {
                grads[i] = grads[i].clipByValue(-1, 1)
              }
              this.ac_optimizer.applyGradients(grads)
            })

            this.updateActorWeights()
          }

          tf.dispose(mb_s0)
          tf.dispose(mb_actions)
          tf.dispose(mb_rewards)
          tf.dispose(mb_s1)
          tf.dispose(pred_next_actions)
          tf.dispose(q_pred)

          resolve()
        })
    })
  }

  updateActorWeights() {
    this.updateWeights(this.targetActor, this.trainingActor)
  }

  updateCriticWeights() {
    this.updateWeights(this.targetCritic, this.trainingCritic)
  }

  updateWeights(target_model, training_model) {
    tf.tidy(() => {
      const training_wts = training_model.getWeights()
      const target_wts = target_model.getWeights()
      for (let i = 0; i < training_wts.length; i++) {
        target_wts[i] = target_wts[i]
          .mul(tf.scalar(1 - global.tau))
          .add(training_wts[i].mul(tf.scalar(global.tau)))
      }
      target_model.setWeights(target_wts)
    })
  }
}
